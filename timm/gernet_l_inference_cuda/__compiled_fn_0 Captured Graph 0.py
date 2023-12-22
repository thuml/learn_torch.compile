from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___stem_conv(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___stem_bn_running_mean = self.L__mod___stem_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_bn_running_var = self.L__mod___stem_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_bn_weight = self.L__mod___stem_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_bn_bias = self.L__mod___stem_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___stem_bn_running_mean, l__mod___stem_bn_running_var, l__mod___stem_bn_weight, l__mod___stem_bn_bias, False, 0.1, 1e-05);  x = l__mod___stem_bn_running_mean = l__mod___stem_bn_running_var = l__mod___stem_bn_weight = l__mod___stem_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___stem_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___stem_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_6 = self.getattr_getattr_L__mod___stages___0_____0___conv1_kxk_conv(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv1_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv1_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv1_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv1_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_bias, False, 0.1, 1e-05);  x_6 = getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv1_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.getattr_getattr_L__mod___stages___0_____0___conv1_kxk_bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_11 = self.getattr_getattr_L__mod___stages___0_____0___conv1_kxk_bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_12 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_conv(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_13 = torch.nn.functional.batch_norm(x_12, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_12 = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_14 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_drop(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_17 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:264, code: x = self.attn(x)
    x_18 = self.getattr_getattr_L__mod___stages___0_____0___attn(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:265, code: x = self.drop_path(x)
    x_19 = self.getattr_getattr_L__mod___stages___0_____0___drop_path(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_20 = self.getattr_getattr_L__mod___stages___0_____0___shortcut_conv(shortcut);  shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_21 = torch.nn.functional.batch_norm(x_20, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_20 = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_22 = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_drop(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_24 = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_act(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:267, code: x = x + self.shortcut(shortcut)
    x_25 = x_19 + x_24;  x_19 = x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    shortcut_1 = self.getattr_getattr_L__mod___stages___0_____0___act(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_26 = self.getattr_getattr_L__mod___stages___1_____0___conv1_kxk_conv(shortcut_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv1_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv1_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv1_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv1_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_27 = torch.nn.functional.batch_norm(x_26, getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_bias, False, 0.1, 1e-05);  x_26 = getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv1_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_28 = self.getattr_getattr_L__mod___stages___1_____0___conv1_kxk_bn_drop(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_31 = self.getattr_getattr_L__mod___stages___1_____0___conv1_kxk_bn_act(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_32 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_conv(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_33 = torch.nn.functional.batch_norm(x_32, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_32 = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_34 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_drop(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_37 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_act(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:264, code: x = self.attn(x)
    x_38 = self.getattr_getattr_L__mod___stages___1_____0___attn(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:265, code: x = self.drop_path(x)
    x_39 = self.getattr_getattr_L__mod___stages___1_____0___drop_path(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_40 = self.getattr_getattr_L__mod___stages___1_____0___shortcut_conv(shortcut_1);  shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_41 = torch.nn.functional.batch_norm(x_40, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_40 = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_42 = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_drop(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_44 = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_act(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:267, code: x = x + self.shortcut(shortcut)
    x_45 = x_39 + x_44;  x_39 = x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___stages___1_____0___act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_46 = self.getattr_getattr_L__mod___stages___1_____1___conv1_kxk_conv(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv1_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv1_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv1_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv1_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_47 = torch.nn.functional.batch_norm(x_46, getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_bias, False, 0.1, 1e-05);  x_46 = getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv1_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_48 = self.getattr_getattr_L__mod___stages___1_____1___conv1_kxk_bn_drop(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_51 = self.getattr_getattr_L__mod___stages___1_____1___conv1_kxk_bn_act(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_52 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_conv(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_53 = torch.nn.functional.batch_norm(x_52, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_52 = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_54 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_drop(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_57 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_act(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:264, code: x = self.attn(x)
    x_58 = self.getattr_getattr_L__mod___stages___1_____1___attn(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:265, code: x = self.drop_path(x)
    x_59 = self.getattr_getattr_L__mod___stages___1_____1___drop_path(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:267, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1_____1___shortcut = self.getattr_getattr_L__mod___stages___1_____1___shortcut(shortcut_2);  shortcut_2 = None
    x_60 = x_59 + getattr_getattr_l__mod___stages___1_____1___shortcut;  x_59 = getattr_getattr_l__mod___stages___1_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    shortcut_3 = self.getattr_getattr_L__mod___stages___1_____1___act(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_61 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_conv(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_62 = torch.nn.functional.batch_norm(x_61, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_61 = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_63 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_drop(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_66 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_act(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_67 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_conv(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_68 = torch.nn.functional.batch_norm(x_67, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_67 = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_69 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_drop(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_72 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_73 = self.getattr_getattr_L__mod___stages___2_____0___conv2b_kxk(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_74 = self.getattr_getattr_L__mod___stages___2_____0___attn(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_75 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_conv(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_76 = torch.nn.functional.batch_norm(x_75, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_75 = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_77 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_drop(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_80 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_act(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_81 = self.getattr_getattr_L__mod___stages___2_____0___attn_last(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_82 = self.getattr_getattr_L__mod___stages___2_____0___drop_path(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_83 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_conv(shortcut_3);  shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_84 = torch.nn.functional.batch_norm(x_83, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_83 = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_85 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_drop(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_87 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_act(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    x_88 = x_82 + x_87;  x_82 = x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___stages___2_____0___act(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_89 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_conv(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_90 = torch.nn.functional.batch_norm(x_89, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_89 = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_91 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_drop(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_94 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_act(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_95 = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_conv(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_96 = torch.nn.functional.batch_norm(x_95, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_95 = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_97 = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_drop(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_100 = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_act(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_101 = self.getattr_getattr_L__mod___stages___2_____1___conv2b_kxk(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_102 = self.getattr_getattr_L__mod___stages___2_____1___attn(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_103 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_conv(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_104 = torch.nn.functional.batch_norm(x_103, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_103 = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_105 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_drop(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_108 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_act(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_109 = self.getattr_getattr_L__mod___stages___2_____1___attn_last(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_110 = self.getattr_getattr_L__mod___stages___2_____1___drop_path(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____1___shortcut = self.getattr_getattr_L__mod___stages___2_____1___shortcut(shortcut_4);  shortcut_4 = None
    x_111 = x_110 + getattr_getattr_l__mod___stages___2_____1___shortcut;  x_110 = getattr_getattr_l__mod___stages___2_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_5 = self.getattr_getattr_L__mod___stages___2_____1___act(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_112 = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_conv(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_113 = torch.nn.functional.batch_norm(x_112, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_112 = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_114 = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_drop(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_117 = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_act(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_118 = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk_conv(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_119 = torch.nn.functional.batch_norm(x_118, getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_118 = getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____2___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_120 = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk_bn_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_123 = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk_bn_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_124 = self.getattr_getattr_L__mod___stages___2_____2___conv2b_kxk(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_125 = self.getattr_getattr_L__mod___stages___2_____2___attn(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_126 = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_conv(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_127 = torch.nn.functional.batch_norm(x_126, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_126 = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_128 = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_drop(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_131 = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_act(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_132 = self.getattr_getattr_L__mod___stages___2_____2___attn_last(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_133 = self.getattr_getattr_L__mod___stages___2_____2___drop_path(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____2___shortcut = self.getattr_getattr_L__mod___stages___2_____2___shortcut(shortcut_5);  shortcut_5 = None
    x_134 = x_133 + getattr_getattr_l__mod___stages___2_____2___shortcut;  x_133 = getattr_getattr_l__mod___stages___2_____2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_6 = self.getattr_getattr_L__mod___stages___2_____2___act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_135 = self.getattr_getattr_L__mod___stages___2_____3___conv1_1x1_conv(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____3___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____3___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____3___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____3___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_136 = torch.nn.functional.batch_norm(x_135, getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_135 = getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____3___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_137 = self.getattr_getattr_L__mod___stages___2_____3___conv1_1x1_bn_drop(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_140 = self.getattr_getattr_L__mod___stages___2_____3___conv1_1x1_bn_act(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_141 = self.getattr_getattr_L__mod___stages___2_____3___conv2_kxk_conv(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____3___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____3___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____3___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____3___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_142 = torch.nn.functional.batch_norm(x_141, getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_141 = getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____3___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_143 = self.getattr_getattr_L__mod___stages___2_____3___conv2_kxk_bn_drop(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_146 = self.getattr_getattr_L__mod___stages___2_____3___conv2_kxk_bn_act(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_147 = self.getattr_getattr_L__mod___stages___2_____3___conv2b_kxk(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_148 = self.getattr_getattr_L__mod___stages___2_____3___attn(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_149 = self.getattr_getattr_L__mod___stages___2_____3___conv3_1x1_conv(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____3___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____3___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____3___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____3___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_150 = torch.nn.functional.batch_norm(x_149, getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_149 = getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____3___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_151 = self.getattr_getattr_L__mod___stages___2_____3___conv3_1x1_bn_drop(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_154 = self.getattr_getattr_L__mod___stages___2_____3___conv3_1x1_bn_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_155 = self.getattr_getattr_L__mod___stages___2_____3___attn_last(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_156 = self.getattr_getattr_L__mod___stages___2_____3___drop_path(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____3___shortcut = self.getattr_getattr_L__mod___stages___2_____3___shortcut(shortcut_6);  shortcut_6 = None
    x_157 = x_156 + getattr_getattr_l__mod___stages___2_____3___shortcut;  x_156 = getattr_getattr_l__mod___stages___2_____3___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_7 = self.getattr_getattr_L__mod___stages___2_____3___act(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_158 = self.getattr_getattr_L__mod___stages___2_____4___conv1_1x1_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____4___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____4___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____4___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____4___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_159 = torch.nn.functional.batch_norm(x_158, getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_158 = getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____4___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_160 = self.getattr_getattr_L__mod___stages___2_____4___conv1_1x1_bn_drop(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_163 = self.getattr_getattr_L__mod___stages___2_____4___conv1_1x1_bn_act(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_164 = self.getattr_getattr_L__mod___stages___2_____4___conv2_kxk_conv(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____4___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____4___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____4___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____4___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_165 = torch.nn.functional.batch_norm(x_164, getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_164 = getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____4___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_166 = self.getattr_getattr_L__mod___stages___2_____4___conv2_kxk_bn_drop(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_169 = self.getattr_getattr_L__mod___stages___2_____4___conv2_kxk_bn_act(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_170 = self.getattr_getattr_L__mod___stages___2_____4___conv2b_kxk(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_171 = self.getattr_getattr_L__mod___stages___2_____4___attn(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_172 = self.getattr_getattr_L__mod___stages___2_____4___conv3_1x1_conv(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____4___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____4___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____4___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____4___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_173 = torch.nn.functional.batch_norm(x_172, getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_172 = getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____4___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_174 = self.getattr_getattr_L__mod___stages___2_____4___conv3_1x1_bn_drop(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_177 = self.getattr_getattr_L__mod___stages___2_____4___conv3_1x1_bn_act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_178 = self.getattr_getattr_L__mod___stages___2_____4___attn_last(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_179 = self.getattr_getattr_L__mod___stages___2_____4___drop_path(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____4___shortcut = self.getattr_getattr_L__mod___stages___2_____4___shortcut(shortcut_7);  shortcut_7 = None
    x_180 = x_179 + getattr_getattr_l__mod___stages___2_____4___shortcut;  x_179 = getattr_getattr_l__mod___stages___2_____4___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_8 = self.getattr_getattr_L__mod___stages___2_____4___act(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_181 = self.getattr_getattr_L__mod___stages___2_____5___conv1_1x1_conv(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____5___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____5___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____5___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____5___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_182 = torch.nn.functional.batch_norm(x_181, getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_181 = getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____5___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_183 = self.getattr_getattr_L__mod___stages___2_____5___conv1_1x1_bn_drop(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_186 = self.getattr_getattr_L__mod___stages___2_____5___conv1_1x1_bn_act(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_187 = self.getattr_getattr_L__mod___stages___2_____5___conv2_kxk_conv(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____5___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____5___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____5___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____5___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_188 = torch.nn.functional.batch_norm(x_187, getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_187 = getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____5___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_189 = self.getattr_getattr_L__mod___stages___2_____5___conv2_kxk_bn_drop(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_192 = self.getattr_getattr_L__mod___stages___2_____5___conv2_kxk_bn_act(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_193 = self.getattr_getattr_L__mod___stages___2_____5___conv2b_kxk(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_194 = self.getattr_getattr_L__mod___stages___2_____5___attn(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_195 = self.getattr_getattr_L__mod___stages___2_____5___conv3_1x1_conv(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____5___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____5___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____5___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____5___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_196 = torch.nn.functional.batch_norm(x_195, getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_195 = getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____5___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_197 = self.getattr_getattr_L__mod___stages___2_____5___conv3_1x1_bn_drop(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_200 = self.getattr_getattr_L__mod___stages___2_____5___conv3_1x1_bn_act(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_201 = self.getattr_getattr_L__mod___stages___2_____5___attn_last(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_202 = self.getattr_getattr_L__mod___stages___2_____5___drop_path(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____5___shortcut = self.getattr_getattr_L__mod___stages___2_____5___shortcut(shortcut_8);  shortcut_8 = None
    x_203 = x_202 + getattr_getattr_l__mod___stages___2_____5___shortcut;  x_202 = getattr_getattr_l__mod___stages___2_____5___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_9 = self.getattr_getattr_L__mod___stages___2_____5___act(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_204 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_conv(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_205 = torch.nn.functional.batch_norm(x_204, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_204 = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_206 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_drop(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_209 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_act(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_210 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_conv(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_211 = torch.nn.functional.batch_norm(x_210, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_210 = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_212 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_drop(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_215 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_act(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_216 = self.getattr_getattr_L__mod___stages___3_____0___conv2b_kxk(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_217 = self.getattr_getattr_L__mod___stages___3_____0___attn(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_218 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_conv(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_219 = torch.nn.functional.batch_norm(x_218, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_218 = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_220 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_drop(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_223 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_act(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_224 = self.getattr_getattr_L__mod___stages___3_____0___attn_last(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_225 = self.getattr_getattr_L__mod___stages___3_____0___drop_path(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_226 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_conv(shortcut_9);  shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_227 = torch.nn.functional.batch_norm(x_226, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_226 = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_228 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_drop(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_230 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_act(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    x_231 = x_225 + x_230;  x_225 = x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_10 = self.getattr_getattr_L__mod___stages___3_____0___act(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_232 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_conv(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_233 = torch.nn.functional.batch_norm(x_232, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_232 = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_234 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_drop(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_237 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_act(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_238 = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk_conv(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_239 = torch.nn.functional.batch_norm(x_238, getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_238 = getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_240 = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk_bn_drop(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_243 = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk_bn_act(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_244 = self.getattr_getattr_L__mod___stages___3_____1___conv2b_kxk(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_245 = self.getattr_getattr_L__mod___stages___3_____1___attn(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_246 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_conv(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_247 = torch.nn.functional.batch_norm(x_246, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_246 = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_248 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_drop(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_251 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_act(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_252 = self.getattr_getattr_L__mod___stages___3_____1___attn_last(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_253 = self.getattr_getattr_L__mod___stages___3_____1___drop_path(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3_____1___shortcut = self.getattr_getattr_L__mod___stages___3_____1___shortcut(shortcut_10);  shortcut_10 = None
    x_254 = x_253 + getattr_getattr_l__mod___stages___3_____1___shortcut;  x_253 = getattr_getattr_l__mod___stages___3_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_11 = self.getattr_getattr_L__mod___stages___3_____1___act(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_255 = self.getattr_getattr_L__mod___stages___3_____2___conv1_1x1_conv(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____2___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____2___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____2___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____2___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_256 = torch.nn.functional.batch_norm(x_255, getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_255 = getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____2___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_257 = self.getattr_getattr_L__mod___stages___3_____2___conv1_1x1_bn_drop(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_260 = self.getattr_getattr_L__mod___stages___3_____2___conv1_1x1_bn_act(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_261 = self.getattr_getattr_L__mod___stages___3_____2___conv2_kxk_conv(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____2___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____2___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____2___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____2___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_262 = torch.nn.functional.batch_norm(x_261, getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_261 = getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____2___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_263 = self.getattr_getattr_L__mod___stages___3_____2___conv2_kxk_bn_drop(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_266 = self.getattr_getattr_L__mod___stages___3_____2___conv2_kxk_bn_act(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_267 = self.getattr_getattr_L__mod___stages___3_____2___conv2b_kxk(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_268 = self.getattr_getattr_L__mod___stages___3_____2___attn(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_269 = self.getattr_getattr_L__mod___stages___3_____2___conv3_1x1_conv(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____2___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____2___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____2___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____2___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_270 = torch.nn.functional.batch_norm(x_269, getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_269 = getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____2___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_271 = self.getattr_getattr_L__mod___stages___3_____2___conv3_1x1_bn_drop(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_274 = self.getattr_getattr_L__mod___stages___3_____2___conv3_1x1_bn_act(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_275 = self.getattr_getattr_L__mod___stages___3_____2___attn_last(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_276 = self.getattr_getattr_L__mod___stages___3_____2___drop_path(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3_____2___shortcut = self.getattr_getattr_L__mod___stages___3_____2___shortcut(shortcut_11);  shortcut_11 = None
    x_277 = x_276 + getattr_getattr_l__mod___stages___3_____2___shortcut;  x_276 = getattr_getattr_l__mod___stages___3_____2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_12 = self.getattr_getattr_L__mod___stages___3_____2___act(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_278 = self.getattr_getattr_L__mod___stages___3_____3___conv1_1x1_conv(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____3___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____3___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____3___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____3___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_279 = torch.nn.functional.batch_norm(x_278, getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_278 = getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____3___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_280 = self.getattr_getattr_L__mod___stages___3_____3___conv1_1x1_bn_drop(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_283 = self.getattr_getattr_L__mod___stages___3_____3___conv1_1x1_bn_act(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_284 = self.getattr_getattr_L__mod___stages___3_____3___conv2_kxk_conv(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____3___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____3___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____3___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____3___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_285 = torch.nn.functional.batch_norm(x_284, getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_284 = getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____3___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_286 = self.getattr_getattr_L__mod___stages___3_____3___conv2_kxk_bn_drop(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_289 = self.getattr_getattr_L__mod___stages___3_____3___conv2_kxk_bn_act(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_290 = self.getattr_getattr_L__mod___stages___3_____3___conv2b_kxk(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_291 = self.getattr_getattr_L__mod___stages___3_____3___attn(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_292 = self.getattr_getattr_L__mod___stages___3_____3___conv3_1x1_conv(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____3___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____3___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____3___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____3___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_293 = torch.nn.functional.batch_norm(x_292, getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_292 = getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____3___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_294 = self.getattr_getattr_L__mod___stages___3_____3___conv3_1x1_bn_drop(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_297 = self.getattr_getattr_L__mod___stages___3_____3___conv3_1x1_bn_act(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_298 = self.getattr_getattr_L__mod___stages___3_____3___attn_last(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_299 = self.getattr_getattr_L__mod___stages___3_____3___drop_path(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3_____3___shortcut = self.getattr_getattr_L__mod___stages___3_____3___shortcut(shortcut_12);  shortcut_12 = None
    x_300 = x_299 + getattr_getattr_l__mod___stages___3_____3___shortcut;  x_299 = getattr_getattr_l__mod___stages___3_____3___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_13 = self.getattr_getattr_L__mod___stages___3_____3___act(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_301 = self.getattr_getattr_L__mod___stages___3_____4___conv1_1x1_conv(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____4___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____4___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____4___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____4___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_302 = torch.nn.functional.batch_norm(x_301, getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_301 = getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____4___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_303 = self.getattr_getattr_L__mod___stages___3_____4___conv1_1x1_bn_drop(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_306 = self.getattr_getattr_L__mod___stages___3_____4___conv1_1x1_bn_act(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_307 = self.getattr_getattr_L__mod___stages___3_____4___conv2_kxk_conv(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____4___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____4___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____4___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____4___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_308 = torch.nn.functional.batch_norm(x_307, getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_307 = getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____4___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_309 = self.getattr_getattr_L__mod___stages___3_____4___conv2_kxk_bn_drop(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_312 = self.getattr_getattr_L__mod___stages___3_____4___conv2_kxk_bn_act(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_313 = self.getattr_getattr_L__mod___stages___3_____4___conv2b_kxk(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_314 = self.getattr_getattr_L__mod___stages___3_____4___attn(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_315 = self.getattr_getattr_L__mod___stages___3_____4___conv3_1x1_conv(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____4___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____4___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____4___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____4___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_316 = torch.nn.functional.batch_norm(x_315, getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_315 = getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____4___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_317 = self.getattr_getattr_L__mod___stages___3_____4___conv3_1x1_bn_drop(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_320 = self.getattr_getattr_L__mod___stages___3_____4___conv3_1x1_bn_act(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_321 = self.getattr_getattr_L__mod___stages___3_____4___attn_last(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_322 = self.getattr_getattr_L__mod___stages___3_____4___drop_path(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3_____4___shortcut = self.getattr_getattr_L__mod___stages___3_____4___shortcut(shortcut_13);  shortcut_13 = None
    x_323 = x_322 + getattr_getattr_l__mod___stages___3_____4___shortcut;  x_322 = getattr_getattr_l__mod___stages___3_____4___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_14 = self.getattr_getattr_L__mod___stages___3_____4___act(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_324 = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_conv(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_325 = torch.nn.functional.batch_norm(x_324, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_324 = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_326 = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_drop(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_329 = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_act(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_330 = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_conv(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_331 = torch.nn.functional.batch_norm(x_330, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_330 = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_332 = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_drop(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_335 = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_act(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_336 = self.getattr_getattr_L__mod___stages___4_____0___conv2b_kxk(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_337 = self.getattr_getattr_L__mod___stages___4_____0___attn(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_338 = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_conv(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_339 = torch.nn.functional.batch_norm(x_338, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_338 = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_340 = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_drop(x_339);  x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_343 = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_act(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_344 = self.getattr_getattr_L__mod___stages___4_____0___attn_last(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_345 = self.getattr_getattr_L__mod___stages___4_____0___drop_path(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___4_____0___shortcut = self.getattr_getattr_L__mod___stages___4_____0___shortcut(shortcut_14);  shortcut_14 = None
    x_346 = x_345 + getattr_getattr_l__mod___stages___4_____0___shortcut;  x_345 = getattr_getattr_l__mod___stages___4_____0___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_15 = self.getattr_getattr_L__mod___stages___4_____0___act(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_347 = self.getattr_getattr_L__mod___stages___4_____1___conv1_1x1_conv(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_348 = torch.nn.functional.batch_norm(x_347, getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_347 = getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_349 = self.getattr_getattr_L__mod___stages___4_____1___conv1_1x1_bn_drop(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_352 = self.getattr_getattr_L__mod___stages___4_____1___conv1_1x1_bn_act(x_349);  x_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_353 = self.getattr_getattr_L__mod___stages___4_____1___conv2_kxk_conv(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___4_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___4_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___4_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_354 = torch.nn.functional.batch_norm(x_353, getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_353 = getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___4_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_355 = self.getattr_getattr_L__mod___stages___4_____1___conv2_kxk_bn_drop(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_358 = self.getattr_getattr_L__mod___stages___4_____1___conv2_kxk_bn_act(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_359 = self.getattr_getattr_L__mod___stages___4_____1___conv2b_kxk(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_360 = self.getattr_getattr_L__mod___stages___4_____1___attn(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_361 = self.getattr_getattr_L__mod___stages___4_____1___conv3_1x1_conv(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_362 = torch.nn.functional.batch_norm(x_361, getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_361 = getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_363 = self.getattr_getattr_L__mod___stages___4_____1___conv3_1x1_bn_drop(x_362);  x_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_366 = self.getattr_getattr_L__mod___stages___4_____1___conv3_1x1_bn_act(x_363);  x_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_367 = self.getattr_getattr_L__mod___stages___4_____1___attn_last(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_368 = self.getattr_getattr_L__mod___stages___4_____1___drop_path(x_367);  x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___4_____1___shortcut = self.getattr_getattr_L__mod___stages___4_____1___shortcut(shortcut_15);  shortcut_15 = None
    x_369 = x_368 + getattr_getattr_l__mod___stages___4_____1___shortcut;  x_368 = getattr_getattr_l__mod___stages___4_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_16 = self.getattr_getattr_L__mod___stages___4_____1___act(x_369);  x_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_370 = self.getattr_getattr_L__mod___stages___4_____2___conv1_1x1_conv(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____2___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____2___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____2___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____2___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_371 = torch.nn.functional.batch_norm(x_370, getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_370 = getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____2___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_372 = self.getattr_getattr_L__mod___stages___4_____2___conv1_1x1_bn_drop(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_375 = self.getattr_getattr_L__mod___stages___4_____2___conv1_1x1_bn_act(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_376 = self.getattr_getattr_L__mod___stages___4_____2___conv2_kxk_conv(x_375);  x_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____2___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___4_____2___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___4_____2___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___4_____2___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_377 = torch.nn.functional.batch_norm(x_376, getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_376 = getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___4_____2___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_378 = self.getattr_getattr_L__mod___stages___4_____2___conv2_kxk_bn_drop(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_381 = self.getattr_getattr_L__mod___stages___4_____2___conv2_kxk_bn_act(x_378);  x_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_382 = self.getattr_getattr_L__mod___stages___4_____2___conv2b_kxk(x_381);  x_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_383 = self.getattr_getattr_L__mod___stages___4_____2___attn(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_384 = self.getattr_getattr_L__mod___stages___4_____2___conv3_1x1_conv(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____2___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____2___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____2___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____2___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_385 = torch.nn.functional.batch_norm(x_384, getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_384 = getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____2___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_386 = self.getattr_getattr_L__mod___stages___4_____2___conv3_1x1_bn_drop(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_389 = self.getattr_getattr_L__mod___stages___4_____2___conv3_1x1_bn_act(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_390 = self.getattr_getattr_L__mod___stages___4_____2___attn_last(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_391 = self.getattr_getattr_L__mod___stages___4_____2___drop_path(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___4_____2___shortcut = self.getattr_getattr_L__mod___stages___4_____2___shortcut(shortcut_16);  shortcut_16 = None
    x_392 = x_391 + getattr_getattr_l__mod___stages___4_____2___shortcut;  x_391 = getattr_getattr_l__mod___stages___4_____2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_17 = self.getattr_getattr_L__mod___stages___4_____2___act(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_393 = self.getattr_getattr_L__mod___stages___4_____3___conv1_1x1_conv(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____3___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____3___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____3___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____3___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_394 = torch.nn.functional.batch_norm(x_393, getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_393 = getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____3___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_395 = self.getattr_getattr_L__mod___stages___4_____3___conv1_1x1_bn_drop(x_394);  x_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_398 = self.getattr_getattr_L__mod___stages___4_____3___conv1_1x1_bn_act(x_395);  x_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_399 = self.getattr_getattr_L__mod___stages___4_____3___conv2_kxk_conv(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____3___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___4_____3___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___4_____3___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___4_____3___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_400 = torch.nn.functional.batch_norm(x_399, getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_399 = getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___4_____3___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_401 = self.getattr_getattr_L__mod___stages___4_____3___conv2_kxk_bn_drop(x_400);  x_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_404 = self.getattr_getattr_L__mod___stages___4_____3___conv2_kxk_bn_act(x_401);  x_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_405 = self.getattr_getattr_L__mod___stages___4_____3___conv2b_kxk(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_406 = self.getattr_getattr_L__mod___stages___4_____3___attn(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_407 = self.getattr_getattr_L__mod___stages___4_____3___conv3_1x1_conv(x_406);  x_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____3___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____3___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____3___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____3___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_408 = torch.nn.functional.batch_norm(x_407, getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_407 = getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____3___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_409 = self.getattr_getattr_L__mod___stages___4_____3___conv3_1x1_bn_drop(x_408);  x_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_412 = self.getattr_getattr_L__mod___stages___4_____3___conv3_1x1_bn_act(x_409);  x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_413 = self.getattr_getattr_L__mod___stages___4_____3___attn_last(x_412);  x_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_414 = self.getattr_getattr_L__mod___stages___4_____3___drop_path(x_413);  x_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___4_____3___shortcut = self.getattr_getattr_L__mod___stages___4_____3___shortcut(shortcut_17);  shortcut_17 = None
    x_415 = x_414 + getattr_getattr_l__mod___stages___4_____3___shortcut;  x_414 = getattr_getattr_l__mod___stages___4_____3___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    x_416 = self.getattr_getattr_L__mod___stages___4_____3___act(x_415);  x_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_417 = self.L__mod___final_conv_conv(x_416);  x_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___final_conv_bn_running_mean = self.L__mod___final_conv_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___final_conv_bn_running_var = self.L__mod___final_conv_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___final_conv_bn_weight = self.L__mod___final_conv_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___final_conv_bn_bias = self.L__mod___final_conv_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_418 = torch.nn.functional.batch_norm(x_417, l__mod___final_conv_bn_running_mean, l__mod___final_conv_bn_running_var, l__mod___final_conv_bn_weight, l__mod___final_conv_bn_bias, False, 0.1, 1e-05);  x_417 = l__mod___final_conv_bn_running_mean = l__mod___final_conv_bn_running_var = l__mod___final_conv_bn_weight = l__mod___final_conv_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_419 = self.L__mod___final_conv_bn_drop(x_418);  x_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_423 = self.L__mod___final_conv_bn_act(x_419);  x_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_424 = self.L__mod___head_global_pool_pool(x_423);  x_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_426 = self.L__mod___head_global_pool_flatten(x_424);  x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_427 = self.L__mod___head_drop(x_426);  x_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_428 = self.L__mod___head_fc(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_429 = self.L__mod___head_flatten(x_428);  x_428 = None
    return (x_429,)
    