from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___stem_conv_1x1_conv(l_inputs_0_)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___stem_conv_1x1_bn_running_mean = self.L__mod___stem_conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv_1x1_bn_running_var = self.L__mod___stem_conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv_1x1_bn_weight = self.L__mod___stem_conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv_1x1_bn_bias = self.L__mod___stem_conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___stem_conv_1x1_bn_running_mean, l__mod___stem_conv_1x1_bn_running_var, l__mod___stem_conv_1x1_bn_weight, l__mod___stem_conv_1x1_bn_bias, False, 0.1, 1e-05);  x = l__mod___stem_conv_1x1_bn_running_mean = l__mod___stem_conv_1x1_bn_running_var = l__mod___stem_conv_1x1_bn_weight = l__mod___stem_conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___stem_conv_1x1_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_4 = self.L__mod___stem_conv_1x1_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_5 = self.L__mod___stem_conv_kxk_conv(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___stem_conv_kxk_bn_running_mean = self.L__mod___stem_conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv_kxk_bn_running_var = self.L__mod___stem_conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv_kxk_bn_weight = self.L__mod___stem_conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv_kxk_bn_bias = self.L__mod___stem_conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_6 = torch.nn.functional.batch_norm(x_5, l__mod___stem_conv_kxk_bn_running_mean, l__mod___stem_conv_kxk_bn_running_var, l__mod___stem_conv_kxk_bn_weight, l__mod___stem_conv_kxk_bn_bias, False, 0.1, 1e-05);  x_5 = l__mod___stem_conv_kxk_bn_running_mean = l__mod___stem_conv_kxk_bn_running_var = l__mod___stem_conv_kxk_bn_weight = l__mod___stem_conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_7 = self.L__mod___stem_conv_kxk_bn_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_9 = self.L__mod___stem_conv_kxk_bn_act(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:534, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_10 = x_4 + x_9;  x_4 = x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_11 = self.L__mod___stem_attn(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    x_12 = self.L__mod___stem_act(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_13 = self.getattr_getattr_L__mod___stages___0_____0___conv_1x1_conv(x_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_14 = torch.nn.functional.batch_norm(x_13, getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_13 = getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_15 = self.getattr_getattr_L__mod___stages___0_____0___conv_1x1_bn_drop(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_17 = self.getattr_getattr_L__mod___stages___0_____0___conv_1x1_bn_act(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_18 = self.getattr_getattr_L__mod___stages___0_____0___conv_kxk_conv(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_19 = torch.nn.functional.batch_norm(x_18, getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_18 = getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_20 = self.getattr_getattr_L__mod___stages___0_____0___conv_kxk_bn_drop(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_22 = self.getattr_getattr_L__mod___stages___0_____0___conv_kxk_bn_act(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:534, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_23 = x_17 + x_22;  x_17 = x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_24 = self.getattr_getattr_L__mod___stages___0_____0___attn(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___0_____0___act = self.getattr_getattr_L__mod___stages___0_____0___act(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____1___identity_running_mean = self.getattr_getattr_L__mod___stages___0_____1___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___identity_running_var = self.getattr_getattr_L__mod___stages___0_____1___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___identity_weight = self.getattr_getattr_L__mod___stages___0_____1___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___identity_bias = self.getattr_getattr_L__mod___stages___0_____1___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_25 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___0_____0___act, getattr_getattr_l__mod___stages___0_____1___identity_running_mean, getattr_getattr_l__mod___stages___0_____1___identity_running_var, getattr_getattr_l__mod___stages___0_____1___identity_weight, getattr_getattr_l__mod___stages___0_____1___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___0_____1___identity_running_mean = getattr_getattr_l__mod___stages___0_____1___identity_running_var = getattr_getattr_l__mod___stages___0_____1___identity_weight = getattr_getattr_l__mod___stages___0_____1___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_26 = self.getattr_getattr_L__mod___stages___0_____1___identity_drop(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity = self.getattr_getattr_L__mod___stages___0_____1___identity_act(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_28 = self.getattr_getattr_L__mod___stages___0_____1___conv_1x1_conv(getattr_getattr_l__mod___stages___0_____0___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_29 = torch.nn.functional.batch_norm(x_28, getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_28 = getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_30 = self.getattr_getattr_L__mod___stages___0_____1___conv_1x1_bn_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_32 = self.getattr_getattr_L__mod___stages___0_____1___conv_1x1_bn_act(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_33 = self.getattr_getattr_L__mod___stages___0_____1___conv_kxk_conv(getattr_getattr_l__mod___stages___0_____0___act);  getattr_getattr_l__mod___stages___0_____0___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_34 = torch.nn.functional.batch_norm(x_33, getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_33 = getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_35 = self.getattr_getattr_L__mod___stages___0_____1___conv_kxk_bn_drop(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_37 = self.getattr_getattr_L__mod___stages___0_____1___conv_kxk_bn_act(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_38 = x_32 + x_37;  x_32 = x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_39 = self.getattr_getattr_L__mod___stages___0_____1___drop_path(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_39 += identity;  x_40 = x_39;  x_39 = identity = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_41 = self.getattr_getattr_L__mod___stages___0_____1___attn(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___0_____1___act = self.getattr_getattr_L__mod___stages___0_____1___act(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_42 = self.getattr_getattr_L__mod___stages___1_____0___conv_1x1_conv(getattr_getattr_l__mod___stages___0_____1___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_43 = torch.nn.functional.batch_norm(x_42, getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_42 = getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_44 = self.getattr_getattr_L__mod___stages___1_____0___conv_1x1_bn_drop(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_46 = self.getattr_getattr_L__mod___stages___1_____0___conv_1x1_bn_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_47 = self.getattr_getattr_L__mod___stages___1_____0___conv_kxk_conv(getattr_getattr_l__mod___stages___0_____1___act);  getattr_getattr_l__mod___stages___0_____1___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_48 = torch.nn.functional.batch_norm(x_47, getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_47 = getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_49 = self.getattr_getattr_L__mod___stages___1_____0___conv_kxk_bn_drop(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_51 = self.getattr_getattr_L__mod___stages___1_____0___conv_kxk_bn_act(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:534, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_52 = x_46 + x_51;  x_46 = x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_53 = self.getattr_getattr_L__mod___stages___1_____0___attn(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___1_____0___act = self.getattr_getattr_L__mod___stages___1_____0___act(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___identity_running_mean = self.getattr_getattr_L__mod___stages___1_____1___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___identity_running_var = self.getattr_getattr_L__mod___stages___1_____1___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___identity_weight = self.getattr_getattr_L__mod___stages___1_____1___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___identity_bias = self.getattr_getattr_L__mod___stages___1_____1___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_54 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___1_____0___act, getattr_getattr_l__mod___stages___1_____1___identity_running_mean, getattr_getattr_l__mod___stages___1_____1___identity_running_var, getattr_getattr_l__mod___stages___1_____1___identity_weight, getattr_getattr_l__mod___stages___1_____1___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___1_____1___identity_running_mean = getattr_getattr_l__mod___stages___1_____1___identity_running_var = getattr_getattr_l__mod___stages___1_____1___identity_weight = getattr_getattr_l__mod___stages___1_____1___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_55 = self.getattr_getattr_L__mod___stages___1_____1___identity_drop(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_1 = self.getattr_getattr_L__mod___stages___1_____1___identity_act(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_57 = self.getattr_getattr_L__mod___stages___1_____1___conv_1x1_conv(getattr_getattr_l__mod___stages___1_____0___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_58 = torch.nn.functional.batch_norm(x_57, getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_57 = getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_59 = self.getattr_getattr_L__mod___stages___1_____1___conv_1x1_bn_drop(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_61 = self.getattr_getattr_L__mod___stages___1_____1___conv_1x1_bn_act(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_62 = self.getattr_getattr_L__mod___stages___1_____1___conv_kxk_conv(getattr_getattr_l__mod___stages___1_____0___act);  getattr_getattr_l__mod___stages___1_____0___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_63 = torch.nn.functional.batch_norm(x_62, getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_62 = getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_64 = self.getattr_getattr_L__mod___stages___1_____1___conv_kxk_bn_drop(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_66 = self.getattr_getattr_L__mod___stages___1_____1___conv_kxk_bn_act(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_67 = x_61 + x_66;  x_61 = x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_68 = self.getattr_getattr_L__mod___stages___1_____1___drop_path(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_68 += identity_1;  x_69 = x_68;  x_68 = identity_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_70 = self.getattr_getattr_L__mod___stages___1_____1___attn(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___1_____1___act = self.getattr_getattr_L__mod___stages___1_____1___act(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____2___identity_running_mean = self.getattr_getattr_L__mod___stages___1_____2___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___identity_running_var = self.getattr_getattr_L__mod___stages___1_____2___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___identity_weight = self.getattr_getattr_L__mod___stages___1_____2___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___identity_bias = self.getattr_getattr_L__mod___stages___1_____2___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_71 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___1_____1___act, getattr_getattr_l__mod___stages___1_____2___identity_running_mean, getattr_getattr_l__mod___stages___1_____2___identity_running_var, getattr_getattr_l__mod___stages___1_____2___identity_weight, getattr_getattr_l__mod___stages___1_____2___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___1_____2___identity_running_mean = getattr_getattr_l__mod___stages___1_____2___identity_running_var = getattr_getattr_l__mod___stages___1_____2___identity_weight = getattr_getattr_l__mod___stages___1_____2___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_72 = self.getattr_getattr_L__mod___stages___1_____2___identity_drop(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_2 = self.getattr_getattr_L__mod___stages___1_____2___identity_act(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_74 = self.getattr_getattr_L__mod___stages___1_____2___conv_1x1_conv(getattr_getattr_l__mod___stages___1_____1___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____2___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____2___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____2___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_75 = torch.nn.functional.batch_norm(x_74, getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_74 = getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____2___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_76 = self.getattr_getattr_L__mod___stages___1_____2___conv_1x1_bn_drop(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_78 = self.getattr_getattr_L__mod___stages___1_____2___conv_1x1_bn_act(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_79 = self.getattr_getattr_L__mod___stages___1_____2___conv_kxk_conv(getattr_getattr_l__mod___stages___1_____1___act);  getattr_getattr_l__mod___stages___1_____1___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____2___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____2___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____2___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_80 = torch.nn.functional.batch_norm(x_79, getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_79 = getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____2___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_81 = self.getattr_getattr_L__mod___stages___1_____2___conv_kxk_bn_drop(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_83 = self.getattr_getattr_L__mod___stages___1_____2___conv_kxk_bn_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_84 = x_78 + x_83;  x_78 = x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_85 = self.getattr_getattr_L__mod___stages___1_____2___drop_path(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_85 += identity_2;  x_86 = x_85;  x_85 = identity_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_87 = self.getattr_getattr_L__mod___stages___1_____2___attn(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___1_____2___act = self.getattr_getattr_L__mod___stages___1_____2___act(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____3___identity_running_mean = self.getattr_getattr_L__mod___stages___1_____3___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____3___identity_running_var = self.getattr_getattr_L__mod___stages___1_____3___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____3___identity_weight = self.getattr_getattr_L__mod___stages___1_____3___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____3___identity_bias = self.getattr_getattr_L__mod___stages___1_____3___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_88 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___1_____2___act, getattr_getattr_l__mod___stages___1_____3___identity_running_mean, getattr_getattr_l__mod___stages___1_____3___identity_running_var, getattr_getattr_l__mod___stages___1_____3___identity_weight, getattr_getattr_l__mod___stages___1_____3___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___1_____3___identity_running_mean = getattr_getattr_l__mod___stages___1_____3___identity_running_var = getattr_getattr_l__mod___stages___1_____3___identity_weight = getattr_getattr_l__mod___stages___1_____3___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_89 = self.getattr_getattr_L__mod___stages___1_____3___identity_drop(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_3 = self.getattr_getattr_L__mod___stages___1_____3___identity_act(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_91 = self.getattr_getattr_L__mod___stages___1_____3___conv_1x1_conv(getattr_getattr_l__mod___stages___1_____2___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____3___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____3___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____3___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____3___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_92 = torch.nn.functional.batch_norm(x_91, getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_91 = getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____3___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_93 = self.getattr_getattr_L__mod___stages___1_____3___conv_1x1_bn_drop(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_95 = self.getattr_getattr_L__mod___stages___1_____3___conv_1x1_bn_act(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_96 = self.getattr_getattr_L__mod___stages___1_____3___conv_kxk_conv(getattr_getattr_l__mod___stages___1_____2___act);  getattr_getattr_l__mod___stages___1_____2___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____3___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____3___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____3___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____3___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_97 = torch.nn.functional.batch_norm(x_96, getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_96 = getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____3___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_98 = self.getattr_getattr_L__mod___stages___1_____3___conv_kxk_bn_drop(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_100 = self.getattr_getattr_L__mod___stages___1_____3___conv_kxk_bn_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_101 = x_95 + x_100;  x_95 = x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_102 = self.getattr_getattr_L__mod___stages___1_____3___drop_path(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_102 += identity_3;  x_103 = x_102;  x_102 = identity_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_104 = self.getattr_getattr_L__mod___stages___1_____3___attn(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___1_____3___act = self.getattr_getattr_L__mod___stages___1_____3___act(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_105 = self.getattr_getattr_L__mod___stages___2_____0___conv_1x1_conv(getattr_getattr_l__mod___stages___1_____3___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_106 = torch.nn.functional.batch_norm(x_105, getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_105 = getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_107 = self.getattr_getattr_L__mod___stages___2_____0___conv_1x1_bn_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_109 = self.getattr_getattr_L__mod___stages___2_____0___conv_1x1_bn_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_110 = self.getattr_getattr_L__mod___stages___2_____0___conv_kxk_conv(getattr_getattr_l__mod___stages___1_____3___act);  getattr_getattr_l__mod___stages___1_____3___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_111 = torch.nn.functional.batch_norm(x_110, getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_110 = getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_112 = self.getattr_getattr_L__mod___stages___2_____0___conv_kxk_bn_drop(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_114 = self.getattr_getattr_L__mod___stages___2_____0___conv_kxk_bn_act(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:534, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_115 = x_109 + x_114;  x_109 = x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_116 = self.getattr_getattr_L__mod___stages___2_____0___attn(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____0___act = self.getattr_getattr_L__mod___stages___2_____0___act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____1___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___identity_running_var = self.getattr_getattr_L__mod___stages___2_____1___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___identity_weight = self.getattr_getattr_L__mod___stages___2_____1___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___identity_bias = self.getattr_getattr_L__mod___stages___2_____1___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_117 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____0___act, getattr_getattr_l__mod___stages___2_____1___identity_running_mean, getattr_getattr_l__mod___stages___2_____1___identity_running_var, getattr_getattr_l__mod___stages___2_____1___identity_weight, getattr_getattr_l__mod___stages___2_____1___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____1___identity_running_mean = getattr_getattr_l__mod___stages___2_____1___identity_running_var = getattr_getattr_l__mod___stages___2_____1___identity_weight = getattr_getattr_l__mod___stages___2_____1___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_118 = self.getattr_getattr_L__mod___stages___2_____1___identity_drop(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_4 = self.getattr_getattr_L__mod___stages___2_____1___identity_act(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_120 = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____0___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_121 = torch.nn.functional.batch_norm(x_120, getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_120 = getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_122 = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1_bn_drop(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_124 = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1_bn_act(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_125 = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____0___act);  getattr_getattr_l__mod___stages___2_____0___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_126 = torch.nn.functional.batch_norm(x_125, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_125 = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_127 = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_drop(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_129 = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_act(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_130 = x_124 + x_129;  x_124 = x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_131 = self.getattr_getattr_L__mod___stages___2_____1___drop_path(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_131 += identity_4;  x_132 = x_131;  x_131 = identity_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_133 = self.getattr_getattr_L__mod___stages___2_____1___attn(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____1___act = self.getattr_getattr_L__mod___stages___2_____1___act(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____2___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___identity_running_var = self.getattr_getattr_L__mod___stages___2_____2___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___identity_weight = self.getattr_getattr_L__mod___stages___2_____2___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___identity_bias = self.getattr_getattr_L__mod___stages___2_____2___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____1___act, getattr_getattr_l__mod___stages___2_____2___identity_running_mean, getattr_getattr_l__mod___stages___2_____2___identity_running_var, getattr_getattr_l__mod___stages___2_____2___identity_weight, getattr_getattr_l__mod___stages___2_____2___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____2___identity_running_mean = getattr_getattr_l__mod___stages___2_____2___identity_running_var = getattr_getattr_l__mod___stages___2_____2___identity_weight = getattr_getattr_l__mod___stages___2_____2___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.getattr_getattr_L__mod___stages___2_____2___identity_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_5 = self.getattr_getattr_L__mod___stages___2_____2___identity_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_137 = self.getattr_getattr_L__mod___stages___2_____2___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____1___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____2___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____2___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____2___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_138 = torch.nn.functional.batch_norm(x_137, getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_137 = getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____2___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_139 = self.getattr_getattr_L__mod___stages___2_____2___conv_1x1_bn_drop(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_141 = self.getattr_getattr_L__mod___stages___2_____2___conv_1x1_bn_act(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_142 = self.getattr_getattr_L__mod___stages___2_____2___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____1___act);  getattr_getattr_l__mod___stages___2_____1___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____2___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____2___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____2___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_143 = torch.nn.functional.batch_norm(x_142, getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_142 = getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____2___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_144 = self.getattr_getattr_L__mod___stages___2_____2___conv_kxk_bn_drop(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_146 = self.getattr_getattr_L__mod___stages___2_____2___conv_kxk_bn_act(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_147 = x_141 + x_146;  x_141 = x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_148 = self.getattr_getattr_L__mod___stages___2_____2___drop_path(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_148 += identity_5;  x_149 = x_148;  x_148 = identity_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_150 = self.getattr_getattr_L__mod___stages___2_____2___attn(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____2___act = self.getattr_getattr_L__mod___stages___2_____2___act(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____3___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____3___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____3___identity_running_var = self.getattr_getattr_L__mod___stages___2_____3___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____3___identity_weight = self.getattr_getattr_L__mod___stages___2_____3___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____3___identity_bias = self.getattr_getattr_L__mod___stages___2_____3___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_151 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____2___act, getattr_getattr_l__mod___stages___2_____3___identity_running_mean, getattr_getattr_l__mod___stages___2_____3___identity_running_var, getattr_getattr_l__mod___stages___2_____3___identity_weight, getattr_getattr_l__mod___stages___2_____3___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____3___identity_running_mean = getattr_getattr_l__mod___stages___2_____3___identity_running_var = getattr_getattr_l__mod___stages___2_____3___identity_weight = getattr_getattr_l__mod___stages___2_____3___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_152 = self.getattr_getattr_L__mod___stages___2_____3___identity_drop(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_6 = self.getattr_getattr_L__mod___stages___2_____3___identity_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_154 = self.getattr_getattr_L__mod___stages___2_____3___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____2___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____3___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____3___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____3___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____3___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_155 = torch.nn.functional.batch_norm(x_154, getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_154 = getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____3___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_156 = self.getattr_getattr_L__mod___stages___2_____3___conv_1x1_bn_drop(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_158 = self.getattr_getattr_L__mod___stages___2_____3___conv_1x1_bn_act(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_159 = self.getattr_getattr_L__mod___stages___2_____3___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____2___act);  getattr_getattr_l__mod___stages___2_____2___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____3___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____3___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____3___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____3___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_160 = torch.nn.functional.batch_norm(x_159, getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_159 = getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____3___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_161 = self.getattr_getattr_L__mod___stages___2_____3___conv_kxk_bn_drop(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_163 = self.getattr_getattr_L__mod___stages___2_____3___conv_kxk_bn_act(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_164 = x_158 + x_163;  x_158 = x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_165 = self.getattr_getattr_L__mod___stages___2_____3___drop_path(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_165 += identity_6;  x_166 = x_165;  x_165 = identity_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_167 = self.getattr_getattr_L__mod___stages___2_____3___attn(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____3___act = self.getattr_getattr_L__mod___stages___2_____3___act(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____4___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____4___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____4___identity_running_var = self.getattr_getattr_L__mod___stages___2_____4___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____4___identity_weight = self.getattr_getattr_L__mod___stages___2_____4___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____4___identity_bias = self.getattr_getattr_L__mod___stages___2_____4___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_168 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____3___act, getattr_getattr_l__mod___stages___2_____4___identity_running_mean, getattr_getattr_l__mod___stages___2_____4___identity_running_var, getattr_getattr_l__mod___stages___2_____4___identity_weight, getattr_getattr_l__mod___stages___2_____4___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____4___identity_running_mean = getattr_getattr_l__mod___stages___2_____4___identity_running_var = getattr_getattr_l__mod___stages___2_____4___identity_weight = getattr_getattr_l__mod___stages___2_____4___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_169 = self.getattr_getattr_L__mod___stages___2_____4___identity_drop(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_7 = self.getattr_getattr_L__mod___stages___2_____4___identity_act(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_171 = self.getattr_getattr_L__mod___stages___2_____4___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____3___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____4___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____4___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____4___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____4___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_172 = torch.nn.functional.batch_norm(x_171, getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_171 = getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____4___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_173 = self.getattr_getattr_L__mod___stages___2_____4___conv_1x1_bn_drop(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_175 = self.getattr_getattr_L__mod___stages___2_____4___conv_1x1_bn_act(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_176 = self.getattr_getattr_L__mod___stages___2_____4___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____3___act);  getattr_getattr_l__mod___stages___2_____3___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____4___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____4___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____4___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____4___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_177 = torch.nn.functional.batch_norm(x_176, getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_176 = getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____4___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_178 = self.getattr_getattr_L__mod___stages___2_____4___conv_kxk_bn_drop(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_180 = self.getattr_getattr_L__mod___stages___2_____4___conv_kxk_bn_act(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_181 = x_175 + x_180;  x_175 = x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_182 = self.getattr_getattr_L__mod___stages___2_____4___drop_path(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_182 += identity_7;  x_183 = x_182;  x_182 = identity_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_184 = self.getattr_getattr_L__mod___stages___2_____4___attn(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____4___act = self.getattr_getattr_L__mod___stages___2_____4___act(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____5___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____5___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____5___identity_running_var = self.getattr_getattr_L__mod___stages___2_____5___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____5___identity_weight = self.getattr_getattr_L__mod___stages___2_____5___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____5___identity_bias = self.getattr_getattr_L__mod___stages___2_____5___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_185 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____4___act, getattr_getattr_l__mod___stages___2_____5___identity_running_mean, getattr_getattr_l__mod___stages___2_____5___identity_running_var, getattr_getattr_l__mod___stages___2_____5___identity_weight, getattr_getattr_l__mod___stages___2_____5___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____5___identity_running_mean = getattr_getattr_l__mod___stages___2_____5___identity_running_var = getattr_getattr_l__mod___stages___2_____5___identity_weight = getattr_getattr_l__mod___stages___2_____5___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_186 = self.getattr_getattr_L__mod___stages___2_____5___identity_drop(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_8 = self.getattr_getattr_L__mod___stages___2_____5___identity_act(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_188 = self.getattr_getattr_L__mod___stages___2_____5___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____4___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____5___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____5___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____5___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____5___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_189 = torch.nn.functional.batch_norm(x_188, getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_188 = getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____5___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_190 = self.getattr_getattr_L__mod___stages___2_____5___conv_1x1_bn_drop(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_192 = self.getattr_getattr_L__mod___stages___2_____5___conv_1x1_bn_act(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_193 = self.getattr_getattr_L__mod___stages___2_____5___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____4___act);  getattr_getattr_l__mod___stages___2_____4___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____5___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____5___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____5___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____5___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_194 = torch.nn.functional.batch_norm(x_193, getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_193 = getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____5___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_195 = self.getattr_getattr_L__mod___stages___2_____5___conv_kxk_bn_drop(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_197 = self.getattr_getattr_L__mod___stages___2_____5___conv_kxk_bn_act(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_198 = x_192 + x_197;  x_192 = x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_199 = self.getattr_getattr_L__mod___stages___2_____5___drop_path(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_199 += identity_8;  x_200 = x_199;  x_199 = identity_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_201 = self.getattr_getattr_L__mod___stages___2_____5___attn(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____5___act = self.getattr_getattr_L__mod___stages___2_____5___act(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____6___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____6___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____6___identity_running_var = self.getattr_getattr_L__mod___stages___2_____6___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____6___identity_weight = self.getattr_getattr_L__mod___stages___2_____6___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____6___identity_bias = self.getattr_getattr_L__mod___stages___2_____6___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_202 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____5___act, getattr_getattr_l__mod___stages___2_____6___identity_running_mean, getattr_getattr_l__mod___stages___2_____6___identity_running_var, getattr_getattr_l__mod___stages___2_____6___identity_weight, getattr_getattr_l__mod___stages___2_____6___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____6___identity_running_mean = getattr_getattr_l__mod___stages___2_____6___identity_running_var = getattr_getattr_l__mod___stages___2_____6___identity_weight = getattr_getattr_l__mod___stages___2_____6___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_203 = self.getattr_getattr_L__mod___stages___2_____6___identity_drop(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_9 = self.getattr_getattr_L__mod___stages___2_____6___identity_act(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_205 = self.getattr_getattr_L__mod___stages___2_____6___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____5___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____6___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____6___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____6___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____6___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_206 = torch.nn.functional.batch_norm(x_205, getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_205 = getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____6___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_207 = self.getattr_getattr_L__mod___stages___2_____6___conv_1x1_bn_drop(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_209 = self.getattr_getattr_L__mod___stages___2_____6___conv_1x1_bn_act(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_210 = self.getattr_getattr_L__mod___stages___2_____6___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____5___act);  getattr_getattr_l__mod___stages___2_____5___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____6___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____6___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____6___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____6___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_211 = torch.nn.functional.batch_norm(x_210, getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_210 = getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____6___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_212 = self.getattr_getattr_L__mod___stages___2_____6___conv_kxk_bn_drop(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_214 = self.getattr_getattr_L__mod___stages___2_____6___conv_kxk_bn_act(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_215 = x_209 + x_214;  x_209 = x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_216 = self.getattr_getattr_L__mod___stages___2_____6___drop_path(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_216 += identity_9;  x_217 = x_216;  x_216 = identity_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_218 = self.getattr_getattr_L__mod___stages___2_____6___attn(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____6___act = self.getattr_getattr_L__mod___stages___2_____6___act(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____7___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____7___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____7___identity_running_var = self.getattr_getattr_L__mod___stages___2_____7___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____7___identity_weight = self.getattr_getattr_L__mod___stages___2_____7___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____7___identity_bias = self.getattr_getattr_L__mod___stages___2_____7___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_219 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____6___act, getattr_getattr_l__mod___stages___2_____7___identity_running_mean, getattr_getattr_l__mod___stages___2_____7___identity_running_var, getattr_getattr_l__mod___stages___2_____7___identity_weight, getattr_getattr_l__mod___stages___2_____7___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____7___identity_running_mean = getattr_getattr_l__mod___stages___2_____7___identity_running_var = getattr_getattr_l__mod___stages___2_____7___identity_weight = getattr_getattr_l__mod___stages___2_____7___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_220 = self.getattr_getattr_L__mod___stages___2_____7___identity_drop(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_10 = self.getattr_getattr_L__mod___stages___2_____7___identity_act(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_222 = self.getattr_getattr_L__mod___stages___2_____7___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____6___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____7___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____7___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____7___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____7___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_223 = torch.nn.functional.batch_norm(x_222, getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_222 = getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____7___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_224 = self.getattr_getattr_L__mod___stages___2_____7___conv_1x1_bn_drop(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_226 = self.getattr_getattr_L__mod___stages___2_____7___conv_1x1_bn_act(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_227 = self.getattr_getattr_L__mod___stages___2_____7___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____6___act);  getattr_getattr_l__mod___stages___2_____6___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____7___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____7___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____7___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____7___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_227 = getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____7___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.getattr_getattr_L__mod___stages___2_____7___conv_kxk_bn_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_231 = self.getattr_getattr_L__mod___stages___2_____7___conv_kxk_bn_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_232 = x_226 + x_231;  x_226 = x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_233 = self.getattr_getattr_L__mod___stages___2_____7___drop_path(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_233 += identity_10;  x_234 = x_233;  x_233 = identity_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_235 = self.getattr_getattr_L__mod___stages___2_____7___attn(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____7___act = self.getattr_getattr_L__mod___stages___2_____7___act(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____8___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____8___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____8___identity_running_var = self.getattr_getattr_L__mod___stages___2_____8___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____8___identity_weight = self.getattr_getattr_L__mod___stages___2_____8___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____8___identity_bias = self.getattr_getattr_L__mod___stages___2_____8___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_236 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____7___act, getattr_getattr_l__mod___stages___2_____8___identity_running_mean, getattr_getattr_l__mod___stages___2_____8___identity_running_var, getattr_getattr_l__mod___stages___2_____8___identity_weight, getattr_getattr_l__mod___stages___2_____8___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____8___identity_running_mean = getattr_getattr_l__mod___stages___2_____8___identity_running_var = getattr_getattr_l__mod___stages___2_____8___identity_weight = getattr_getattr_l__mod___stages___2_____8___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_237 = self.getattr_getattr_L__mod___stages___2_____8___identity_drop(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_11 = self.getattr_getattr_L__mod___stages___2_____8___identity_act(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_239 = self.getattr_getattr_L__mod___stages___2_____8___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____7___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____8___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____8___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____8___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____8___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_240 = torch.nn.functional.batch_norm(x_239, getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_239 = getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____8___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_241 = self.getattr_getattr_L__mod___stages___2_____8___conv_1x1_bn_drop(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_243 = self.getattr_getattr_L__mod___stages___2_____8___conv_1x1_bn_act(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_244 = self.getattr_getattr_L__mod___stages___2_____8___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____7___act);  getattr_getattr_l__mod___stages___2_____7___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____8___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____8___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____8___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____8___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_245 = torch.nn.functional.batch_norm(x_244, getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_244 = getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____8___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_246 = self.getattr_getattr_L__mod___stages___2_____8___conv_kxk_bn_drop(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_248 = self.getattr_getattr_L__mod___stages___2_____8___conv_kxk_bn_act(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_249 = x_243 + x_248;  x_243 = x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_250 = self.getattr_getattr_L__mod___stages___2_____8___drop_path(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_250 += identity_11;  x_251 = x_250;  x_250 = identity_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_252 = self.getattr_getattr_L__mod___stages___2_____8___attn(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____8___act = self.getattr_getattr_L__mod___stages___2_____8___act(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____9___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____9___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____9___identity_running_var = self.getattr_getattr_L__mod___stages___2_____9___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____9___identity_weight = self.getattr_getattr_L__mod___stages___2_____9___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____9___identity_bias = self.getattr_getattr_L__mod___stages___2_____9___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_253 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____8___act, getattr_getattr_l__mod___stages___2_____9___identity_running_mean, getattr_getattr_l__mod___stages___2_____9___identity_running_var, getattr_getattr_l__mod___stages___2_____9___identity_weight, getattr_getattr_l__mod___stages___2_____9___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____9___identity_running_mean = getattr_getattr_l__mod___stages___2_____9___identity_running_var = getattr_getattr_l__mod___stages___2_____9___identity_weight = getattr_getattr_l__mod___stages___2_____9___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_254 = self.getattr_getattr_L__mod___stages___2_____9___identity_drop(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_12 = self.getattr_getattr_L__mod___stages___2_____9___identity_act(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_256 = self.getattr_getattr_L__mod___stages___2_____9___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____8___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____9___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____9___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____9___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____9___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_257 = torch.nn.functional.batch_norm(x_256, getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_256 = getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____9___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_258 = self.getattr_getattr_L__mod___stages___2_____9___conv_1x1_bn_drop(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_260 = self.getattr_getattr_L__mod___stages___2_____9___conv_1x1_bn_act(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_261 = self.getattr_getattr_L__mod___stages___2_____9___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____8___act);  getattr_getattr_l__mod___stages___2_____8___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____9___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____9___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____9___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____9___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_262 = torch.nn.functional.batch_norm(x_261, getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_261 = getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____9___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_263 = self.getattr_getattr_L__mod___stages___2_____9___conv_kxk_bn_drop(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_265 = self.getattr_getattr_L__mod___stages___2_____9___conv_kxk_bn_act(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_266 = x_260 + x_265;  x_260 = x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_267 = self.getattr_getattr_L__mod___stages___2_____9___drop_path(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_267 += identity_12;  x_268 = x_267;  x_267 = identity_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_269 = self.getattr_getattr_L__mod___stages___2_____9___attn(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____9___act = self.getattr_getattr_L__mod___stages___2_____9___act(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____10___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____10___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____10___identity_running_var = self.getattr_getattr_L__mod___stages___2_____10___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____10___identity_weight = self.getattr_getattr_L__mod___stages___2_____10___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____10___identity_bias = self.getattr_getattr_L__mod___stages___2_____10___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_270 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____9___act, getattr_getattr_l__mod___stages___2_____10___identity_running_mean, getattr_getattr_l__mod___stages___2_____10___identity_running_var, getattr_getattr_l__mod___stages___2_____10___identity_weight, getattr_getattr_l__mod___stages___2_____10___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____10___identity_running_mean = getattr_getattr_l__mod___stages___2_____10___identity_running_var = getattr_getattr_l__mod___stages___2_____10___identity_weight = getattr_getattr_l__mod___stages___2_____10___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_271 = self.getattr_getattr_L__mod___stages___2_____10___identity_drop(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_13 = self.getattr_getattr_L__mod___stages___2_____10___identity_act(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_273 = self.getattr_getattr_L__mod___stages___2_____10___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____9___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____10___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____10___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____10___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____10___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_274 = torch.nn.functional.batch_norm(x_273, getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_273 = getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____10___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_275 = self.getattr_getattr_L__mod___stages___2_____10___conv_1x1_bn_drop(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_277 = self.getattr_getattr_L__mod___stages___2_____10___conv_1x1_bn_act(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_278 = self.getattr_getattr_L__mod___stages___2_____10___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____9___act);  getattr_getattr_l__mod___stages___2_____9___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____10___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____10___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____10___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____10___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_279 = torch.nn.functional.batch_norm(x_278, getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_278 = getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____10___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_280 = self.getattr_getattr_L__mod___stages___2_____10___conv_kxk_bn_drop(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_282 = self.getattr_getattr_L__mod___stages___2_____10___conv_kxk_bn_act(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_283 = x_277 + x_282;  x_277 = x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_284 = self.getattr_getattr_L__mod___stages___2_____10___drop_path(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_284 += identity_13;  x_285 = x_284;  x_284 = identity_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_286 = self.getattr_getattr_L__mod___stages___2_____10___attn(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____10___act = self.getattr_getattr_L__mod___stages___2_____10___act(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____11___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____11___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____11___identity_running_var = self.getattr_getattr_L__mod___stages___2_____11___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____11___identity_weight = self.getattr_getattr_L__mod___stages___2_____11___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____11___identity_bias = self.getattr_getattr_L__mod___stages___2_____11___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_287 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____10___act, getattr_getattr_l__mod___stages___2_____11___identity_running_mean, getattr_getattr_l__mod___stages___2_____11___identity_running_var, getattr_getattr_l__mod___stages___2_____11___identity_weight, getattr_getattr_l__mod___stages___2_____11___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____11___identity_running_mean = getattr_getattr_l__mod___stages___2_____11___identity_running_var = getattr_getattr_l__mod___stages___2_____11___identity_weight = getattr_getattr_l__mod___stages___2_____11___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_288 = self.getattr_getattr_L__mod___stages___2_____11___identity_drop(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_14 = self.getattr_getattr_L__mod___stages___2_____11___identity_act(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_290 = self.getattr_getattr_L__mod___stages___2_____11___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____10___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____11___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____11___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____11___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____11___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_291 = torch.nn.functional.batch_norm(x_290, getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_290 = getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____11___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_292 = self.getattr_getattr_L__mod___stages___2_____11___conv_1x1_bn_drop(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_294 = self.getattr_getattr_L__mod___stages___2_____11___conv_1x1_bn_act(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_295 = self.getattr_getattr_L__mod___stages___2_____11___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____10___act);  getattr_getattr_l__mod___stages___2_____10___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____11___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____11___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____11___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____11___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_296 = torch.nn.functional.batch_norm(x_295, getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_295 = getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____11___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_297 = self.getattr_getattr_L__mod___stages___2_____11___conv_kxk_bn_drop(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_299 = self.getattr_getattr_L__mod___stages___2_____11___conv_kxk_bn_act(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_300 = x_294 + x_299;  x_294 = x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_301 = self.getattr_getattr_L__mod___stages___2_____11___drop_path(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_301 += identity_14;  x_302 = x_301;  x_301 = identity_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_303 = self.getattr_getattr_L__mod___stages___2_____11___attn(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____11___act = self.getattr_getattr_L__mod___stages___2_____11___act(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____12___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____12___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____12___identity_running_var = self.getattr_getattr_L__mod___stages___2_____12___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____12___identity_weight = self.getattr_getattr_L__mod___stages___2_____12___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____12___identity_bias = self.getattr_getattr_L__mod___stages___2_____12___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_304 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____11___act, getattr_getattr_l__mod___stages___2_____12___identity_running_mean, getattr_getattr_l__mod___stages___2_____12___identity_running_var, getattr_getattr_l__mod___stages___2_____12___identity_weight, getattr_getattr_l__mod___stages___2_____12___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____12___identity_running_mean = getattr_getattr_l__mod___stages___2_____12___identity_running_var = getattr_getattr_l__mod___stages___2_____12___identity_weight = getattr_getattr_l__mod___stages___2_____12___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_305 = self.getattr_getattr_L__mod___stages___2_____12___identity_drop(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_15 = self.getattr_getattr_L__mod___stages___2_____12___identity_act(x_305);  x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_307 = self.getattr_getattr_L__mod___stages___2_____12___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____11___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____12___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____12___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____12___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____12___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_308 = torch.nn.functional.batch_norm(x_307, getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_307 = getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____12___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_309 = self.getattr_getattr_L__mod___stages___2_____12___conv_1x1_bn_drop(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_311 = self.getattr_getattr_L__mod___stages___2_____12___conv_1x1_bn_act(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_312 = self.getattr_getattr_L__mod___stages___2_____12___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____11___act);  getattr_getattr_l__mod___stages___2_____11___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____12___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____12___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____12___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____12___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_313 = torch.nn.functional.batch_norm(x_312, getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_312 = getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____12___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_314 = self.getattr_getattr_L__mod___stages___2_____12___conv_kxk_bn_drop(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_316 = self.getattr_getattr_L__mod___stages___2_____12___conv_kxk_bn_act(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_317 = x_311 + x_316;  x_311 = x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_318 = self.getattr_getattr_L__mod___stages___2_____12___drop_path(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_318 += identity_15;  x_319 = x_318;  x_318 = identity_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_320 = self.getattr_getattr_L__mod___stages___2_____12___attn(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____12___act = self.getattr_getattr_L__mod___stages___2_____12___act(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____13___identity_running_mean = self.getattr_getattr_L__mod___stages___2_____13___identity_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____13___identity_running_var = self.getattr_getattr_L__mod___stages___2_____13___identity_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____13___identity_weight = self.getattr_getattr_L__mod___stages___2_____13___identity_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____13___identity_bias = self.getattr_getattr_L__mod___stages___2_____13___identity_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_321 = torch.nn.functional.batch_norm(getattr_getattr_l__mod___stages___2_____12___act, getattr_getattr_l__mod___stages___2_____13___identity_running_mean, getattr_getattr_l__mod___stages___2_____13___identity_running_var, getattr_getattr_l__mod___stages___2_____13___identity_weight, getattr_getattr_l__mod___stages___2_____13___identity_bias, False, 0.1, 1e-05);  getattr_getattr_l__mod___stages___2_____13___identity_running_mean = getattr_getattr_l__mod___stages___2_____13___identity_running_var = getattr_getattr_l__mod___stages___2_____13___identity_weight = getattr_getattr_l__mod___stages___2_____13___identity_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_322 = self.getattr_getattr_L__mod___stages___2_____13___identity_drop(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    identity_16 = self.getattr_getattr_L__mod___stages___2_____13___identity_act(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_324 = self.getattr_getattr_L__mod___stages___2_____13___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____12___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____13___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____13___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____13___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____13___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_325 = torch.nn.functional.batch_norm(x_324, getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_324 = getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____13___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_326 = self.getattr_getattr_L__mod___stages___2_____13___conv_1x1_bn_drop(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_328 = self.getattr_getattr_L__mod___stages___2_____13___conv_1x1_bn_act(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_329 = self.getattr_getattr_L__mod___stages___2_____13___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____12___act);  getattr_getattr_l__mod___stages___2_____12___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____13___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____13___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____13___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____13___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_330 = torch.nn.functional.batch_norm(x_329, getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_329 = getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____13___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_331 = self.getattr_getattr_L__mod___stages___2_____13___conv_kxk_bn_drop(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_333 = self.getattr_getattr_L__mod___stages___2_____13___conv_kxk_bn_act(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:537, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_334 = x_328 + x_333;  x_328 = x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:538, code: x = self.drop_path(x)  # not in the paper / official impl, experimental
    x_335 = self.getattr_getattr_L__mod___stages___2_____13___drop_path(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:539, code: x += identity
    x_335 += identity_16;  x_336 = x_335;  x_335 = identity_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_337 = self.getattr_getattr_L__mod___stages___2_____13___attn(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    getattr_getattr_l__mod___stages___2_____13___act = self.getattr_getattr_L__mod___stages___2_____13___act(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_338 = self.getattr_getattr_L__mod___stages___3_____0___conv_1x1_conv(getattr_getattr_l__mod___stages___2_____13___act)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_339 = torch.nn.functional.batch_norm(x_338, getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_bias, False, 0.1, 1e-05);  x_338 = getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_340 = self.getattr_getattr_L__mod___stages___3_____0___conv_1x1_bn_drop(x_339);  x_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_342 = self.getattr_getattr_L__mod___stages___3_____0___conv_1x1_bn_act(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_343 = self.getattr_getattr_L__mod___stages___3_____0___conv_kxk_conv(getattr_getattr_l__mod___stages___2_____13___act);  getattr_getattr_l__mod___stages___2_____13___act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_344 = torch.nn.functional.batch_norm(x_343, getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_bias, False, 0.1, 1e-05);  x_343 = getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_345 = self.getattr_getattr_L__mod___stages___3_____0___conv_kxk_bn_drop(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_347 = self.getattr_getattr_L__mod___stages___3_____0___conv_kxk_bn_act(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:534, code: x = self.conv_1x1(x) + self.conv_kxk(x)
    x_348 = x_342 + x_347;  x_342 = x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:540, code: x = self.attn(x)  # no attn in the paper / official impl, experimental
    x_349 = self.getattr_getattr_L__mod___stages___3_____0___attn(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    x_350 = self.getattr_getattr_L__mod___stages___3_____0___act(x_349);  x_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1250, code: x = self.final_conv(x)
    x_352 = self.L__mod___final_conv(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_353 = self.L__mod___head_global_pool_pool(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_355 = self.L__mod___head_global_pool_flatten(x_353);  x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_356 = self.L__mod___head_drop(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_357 = self.L__mod___head_fc(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_358 = self.L__mod___head_flatten(x_357);  x_357 = None
    return (x_358,)
    