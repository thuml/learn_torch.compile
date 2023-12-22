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
    x_6 = self.getattr_L__mod___features___0___conv_dw_conv(shortcut);  shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___0___conv_dw_bn_running_mean = self.getattr_L__mod___features___0___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___0___conv_dw_bn_running_var = self.getattr_L__mod___features___0___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___0___conv_dw_bn_weight = self.getattr_L__mod___features___0___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___0___conv_dw_bn_bias = self.getattr_L__mod___features___0___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, getattr_l__mod___features___0___conv_dw_bn_running_mean, getattr_l__mod___features___0___conv_dw_bn_running_var, getattr_l__mod___features___0___conv_dw_bn_weight, getattr_l__mod___features___0___conv_dw_bn_bias, False, 0.1, 1e-05);  x_6 = getattr_l__mod___features___0___conv_dw_bn_running_mean = getattr_l__mod___features___0___conv_dw_bn_running_var = getattr_l__mod___features___0___conv_dw_bn_weight = getattr_l__mod___features___0___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.getattr_L__mod___features___0___conv_dw_bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_11 = self.getattr_L__mod___features___0___conv_dw_bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_12 = self.getattr_L__mod___features___0___act_dw(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_13 = self.getattr_L__mod___features___0___conv_pwl_conv(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___0___conv_pwl_bn_running_mean = self.getattr_L__mod___features___0___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___0___conv_pwl_bn_running_var = self.getattr_L__mod___features___0___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___0___conv_pwl_bn_weight = self.getattr_L__mod___features___0___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___0___conv_pwl_bn_bias = self.getattr_L__mod___features___0___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_14 = torch.nn.functional.batch_norm(x_13, getattr_l__mod___features___0___conv_pwl_bn_running_mean, getattr_l__mod___features___0___conv_pwl_bn_running_var, getattr_l__mod___features___0___conv_pwl_bn_weight, getattr_l__mod___features___0___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_13 = getattr_l__mod___features___0___conv_pwl_bn_running_mean = getattr_l__mod___features___0___conv_pwl_bn_running_var = getattr_l__mod___features___0___conv_pwl_bn_weight = getattr_l__mod___features___0___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_15 = self.getattr_L__mod___features___0___conv_pwl_bn_drop(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_1 = self.getattr_L__mod___features___0___conv_pwl_bn_act(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_19 = self.getattr_L__mod___features___1___conv_exp_conv(shortcut_1);  shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___1___conv_exp_bn_running_mean = self.getattr_L__mod___features___1___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___1___conv_exp_bn_running_var = self.getattr_L__mod___features___1___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___1___conv_exp_bn_weight = self.getattr_L__mod___features___1___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___1___conv_exp_bn_bias = self.getattr_L__mod___features___1___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_20 = torch.nn.functional.batch_norm(x_19, getattr_l__mod___features___1___conv_exp_bn_running_mean, getattr_l__mod___features___1___conv_exp_bn_running_var, getattr_l__mod___features___1___conv_exp_bn_weight, getattr_l__mod___features___1___conv_exp_bn_bias, False, 0.1, 1e-05);  x_19 = getattr_l__mod___features___1___conv_exp_bn_running_mean = getattr_l__mod___features___1___conv_exp_bn_running_var = getattr_l__mod___features___1___conv_exp_bn_weight = getattr_l__mod___features___1___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_21 = self.getattr_L__mod___features___1___conv_exp_bn_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_24 = self.getattr_L__mod___features___1___conv_exp_bn_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_25 = self.getattr_L__mod___features___1___conv_dw_conv(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___1___conv_dw_bn_running_mean = self.getattr_L__mod___features___1___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___1___conv_dw_bn_running_var = self.getattr_L__mod___features___1___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___1___conv_dw_bn_weight = self.getattr_L__mod___features___1___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___1___conv_dw_bn_bias = self.getattr_L__mod___features___1___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_26 = torch.nn.functional.batch_norm(x_25, getattr_l__mod___features___1___conv_dw_bn_running_mean, getattr_l__mod___features___1___conv_dw_bn_running_var, getattr_l__mod___features___1___conv_dw_bn_weight, getattr_l__mod___features___1___conv_dw_bn_bias, False, 0.1, 1e-05);  x_25 = getattr_l__mod___features___1___conv_dw_bn_running_mean = getattr_l__mod___features___1___conv_dw_bn_running_var = getattr_l__mod___features___1___conv_dw_bn_weight = getattr_l__mod___features___1___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_27 = self.getattr_L__mod___features___1___conv_dw_bn_drop(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_30 = self.getattr_L__mod___features___1___conv_dw_bn_act(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_31 = self.getattr_L__mod___features___1___act_dw(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_32 = self.getattr_L__mod___features___1___conv_pwl_conv(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___1___conv_pwl_bn_running_mean = self.getattr_L__mod___features___1___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___1___conv_pwl_bn_running_var = self.getattr_L__mod___features___1___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___1___conv_pwl_bn_weight = self.getattr_L__mod___features___1___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___1___conv_pwl_bn_bias = self.getattr_L__mod___features___1___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_33 = torch.nn.functional.batch_norm(x_32, getattr_l__mod___features___1___conv_pwl_bn_running_mean, getattr_l__mod___features___1___conv_pwl_bn_running_var, getattr_l__mod___features___1___conv_pwl_bn_weight, getattr_l__mod___features___1___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_32 = getattr_l__mod___features___1___conv_pwl_bn_running_mean = getattr_l__mod___features___1___conv_pwl_bn_running_var = getattr_l__mod___features___1___conv_pwl_bn_weight = getattr_l__mod___features___1___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_34 = self.getattr_L__mod___features___1___conv_pwl_bn_drop(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_2 = self.getattr_L__mod___features___1___conv_pwl_bn_act(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_38 = self.getattr_L__mod___features___2___conv_exp_conv(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___2___conv_exp_bn_running_mean = self.getattr_L__mod___features___2___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___2___conv_exp_bn_running_var = self.getattr_L__mod___features___2___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___2___conv_exp_bn_weight = self.getattr_L__mod___features___2___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___2___conv_exp_bn_bias = self.getattr_L__mod___features___2___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_39 = torch.nn.functional.batch_norm(x_38, getattr_l__mod___features___2___conv_exp_bn_running_mean, getattr_l__mod___features___2___conv_exp_bn_running_var, getattr_l__mod___features___2___conv_exp_bn_weight, getattr_l__mod___features___2___conv_exp_bn_bias, False, 0.1, 1e-05);  x_38 = getattr_l__mod___features___2___conv_exp_bn_running_mean = getattr_l__mod___features___2___conv_exp_bn_running_var = getattr_l__mod___features___2___conv_exp_bn_weight = getattr_l__mod___features___2___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_40 = self.getattr_L__mod___features___2___conv_exp_bn_drop(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_43 = self.getattr_L__mod___features___2___conv_exp_bn_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_44 = self.getattr_L__mod___features___2___conv_dw_conv(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___2___conv_dw_bn_running_mean = self.getattr_L__mod___features___2___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___2___conv_dw_bn_running_var = self.getattr_L__mod___features___2___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___2___conv_dw_bn_weight = self.getattr_L__mod___features___2___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___2___conv_dw_bn_bias = self.getattr_L__mod___features___2___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_45 = torch.nn.functional.batch_norm(x_44, getattr_l__mod___features___2___conv_dw_bn_running_mean, getattr_l__mod___features___2___conv_dw_bn_running_var, getattr_l__mod___features___2___conv_dw_bn_weight, getattr_l__mod___features___2___conv_dw_bn_bias, False, 0.1, 1e-05);  x_44 = getattr_l__mod___features___2___conv_dw_bn_running_mean = getattr_l__mod___features___2___conv_dw_bn_running_var = getattr_l__mod___features___2___conv_dw_bn_weight = getattr_l__mod___features___2___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_46 = self.getattr_L__mod___features___2___conv_dw_bn_drop(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_49 = self.getattr_L__mod___features___2___conv_dw_bn_act(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_50 = self.getattr_L__mod___features___2___act_dw(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_51 = self.getattr_L__mod___features___2___conv_pwl_conv(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___2___conv_pwl_bn_running_mean = self.getattr_L__mod___features___2___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___2___conv_pwl_bn_running_var = self.getattr_L__mod___features___2___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___2___conv_pwl_bn_weight = self.getattr_L__mod___features___2___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___2___conv_pwl_bn_bias = self.getattr_L__mod___features___2___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_52 = torch.nn.functional.batch_norm(x_51, getattr_l__mod___features___2___conv_pwl_bn_running_mean, getattr_l__mod___features___2___conv_pwl_bn_running_var, getattr_l__mod___features___2___conv_pwl_bn_weight, getattr_l__mod___features___2___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_51 = getattr_l__mod___features___2___conv_pwl_bn_running_mean = getattr_l__mod___features___2___conv_pwl_bn_running_var = getattr_l__mod___features___2___conv_pwl_bn_weight = getattr_l__mod___features___2___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_53 = self.getattr_L__mod___features___2___conv_pwl_bn_drop(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_56 = self.getattr_L__mod___features___2___conv_pwl_bn_act(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem = x_56[(slice(None, None, None), slice(0, 27, None))]
    add = getitem + shortcut_2;  getitem = shortcut_2 = None
    getitem_1 = x_56[(slice(None, None, None), slice(27, None, None))];  x_56 = None
    shortcut_3 = torch.cat([add, getitem_1], dim = 1);  add = getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_58 = self.getattr_L__mod___features___3___conv_exp_conv(shortcut_3);  shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___3___conv_exp_bn_running_mean = self.getattr_L__mod___features___3___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___3___conv_exp_bn_running_var = self.getattr_L__mod___features___3___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___3___conv_exp_bn_weight = self.getattr_L__mod___features___3___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___3___conv_exp_bn_bias = self.getattr_L__mod___features___3___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_59 = torch.nn.functional.batch_norm(x_58, getattr_l__mod___features___3___conv_exp_bn_running_mean, getattr_l__mod___features___3___conv_exp_bn_running_var, getattr_l__mod___features___3___conv_exp_bn_weight, getattr_l__mod___features___3___conv_exp_bn_bias, False, 0.1, 1e-05);  x_58 = getattr_l__mod___features___3___conv_exp_bn_running_mean = getattr_l__mod___features___3___conv_exp_bn_running_var = getattr_l__mod___features___3___conv_exp_bn_weight = getattr_l__mod___features___3___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_60 = self.getattr_L__mod___features___3___conv_exp_bn_drop(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_63 = self.getattr_L__mod___features___3___conv_exp_bn_act(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_64 = self.getattr_L__mod___features___3___conv_dw_conv(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___3___conv_dw_bn_running_mean = self.getattr_L__mod___features___3___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___3___conv_dw_bn_running_var = self.getattr_L__mod___features___3___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___3___conv_dw_bn_weight = self.getattr_L__mod___features___3___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___3___conv_dw_bn_bias = self.getattr_L__mod___features___3___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_65 = torch.nn.functional.batch_norm(x_64, getattr_l__mod___features___3___conv_dw_bn_running_mean, getattr_l__mod___features___3___conv_dw_bn_running_var, getattr_l__mod___features___3___conv_dw_bn_weight, getattr_l__mod___features___3___conv_dw_bn_bias, False, 0.1, 1e-05);  x_64 = getattr_l__mod___features___3___conv_dw_bn_running_mean = getattr_l__mod___features___3___conv_dw_bn_running_var = getattr_l__mod___features___3___conv_dw_bn_weight = getattr_l__mod___features___3___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_66 = self.getattr_L__mod___features___3___conv_dw_bn_drop(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_69 = self.getattr_L__mod___features___3___conv_dw_bn_act(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_69.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_1 = self.getattr_L__mod___features___3___se_fc1(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___3___se_bn = self.getattr_L__mod___features___3___se_bn(x_se_1);  x_se_1 = None
    x_se_2 = self.getattr_L__mod___features___3___se_act(getattr_l__mod___features___3___se_bn);  getattr_l__mod___features___3___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_3 = self.getattr_L__mod___features___3___se_fc2(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid = x_se_3.sigmoid();  x_se_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_70 = x_69 * sigmoid;  x_69 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_71 = self.getattr_L__mod___features___3___act_dw(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_72 = self.getattr_L__mod___features___3___conv_pwl_conv(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___3___conv_pwl_bn_running_mean = self.getattr_L__mod___features___3___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___3___conv_pwl_bn_running_var = self.getattr_L__mod___features___3___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___3___conv_pwl_bn_weight = self.getattr_L__mod___features___3___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___3___conv_pwl_bn_bias = self.getattr_L__mod___features___3___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_73 = torch.nn.functional.batch_norm(x_72, getattr_l__mod___features___3___conv_pwl_bn_running_mean, getattr_l__mod___features___3___conv_pwl_bn_running_var, getattr_l__mod___features___3___conv_pwl_bn_weight, getattr_l__mod___features___3___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_72 = getattr_l__mod___features___3___conv_pwl_bn_running_mean = getattr_l__mod___features___3___conv_pwl_bn_running_var = getattr_l__mod___features___3___conv_pwl_bn_weight = getattr_l__mod___features___3___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_74 = self.getattr_L__mod___features___3___conv_pwl_bn_drop(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_4 = self.getattr_L__mod___features___3___conv_pwl_bn_act(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_78 = self.getattr_L__mod___features___4___conv_exp_conv(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___4___conv_exp_bn_running_mean = self.getattr_L__mod___features___4___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___4___conv_exp_bn_running_var = self.getattr_L__mod___features___4___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___4___conv_exp_bn_weight = self.getattr_L__mod___features___4___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___4___conv_exp_bn_bias = self.getattr_L__mod___features___4___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_79 = torch.nn.functional.batch_norm(x_78, getattr_l__mod___features___4___conv_exp_bn_running_mean, getattr_l__mod___features___4___conv_exp_bn_running_var, getattr_l__mod___features___4___conv_exp_bn_weight, getattr_l__mod___features___4___conv_exp_bn_bias, False, 0.1, 1e-05);  x_78 = getattr_l__mod___features___4___conv_exp_bn_running_mean = getattr_l__mod___features___4___conv_exp_bn_running_var = getattr_l__mod___features___4___conv_exp_bn_weight = getattr_l__mod___features___4___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_80 = self.getattr_L__mod___features___4___conv_exp_bn_drop(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_83 = self.getattr_L__mod___features___4___conv_exp_bn_act(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_84 = self.getattr_L__mod___features___4___conv_dw_conv(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___4___conv_dw_bn_running_mean = self.getattr_L__mod___features___4___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___4___conv_dw_bn_running_var = self.getattr_L__mod___features___4___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___4___conv_dw_bn_weight = self.getattr_L__mod___features___4___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___4___conv_dw_bn_bias = self.getattr_L__mod___features___4___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_85 = torch.nn.functional.batch_norm(x_84, getattr_l__mod___features___4___conv_dw_bn_running_mean, getattr_l__mod___features___4___conv_dw_bn_running_var, getattr_l__mod___features___4___conv_dw_bn_weight, getattr_l__mod___features___4___conv_dw_bn_bias, False, 0.1, 1e-05);  x_84 = getattr_l__mod___features___4___conv_dw_bn_running_mean = getattr_l__mod___features___4___conv_dw_bn_running_var = getattr_l__mod___features___4___conv_dw_bn_weight = getattr_l__mod___features___4___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_86 = self.getattr_L__mod___features___4___conv_dw_bn_drop(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_89 = self.getattr_L__mod___features___4___conv_dw_bn_act(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_89.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_5 = self.getattr_L__mod___features___4___se_fc1(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___4___se_bn = self.getattr_L__mod___features___4___se_bn(x_se_5);  x_se_5 = None
    x_se_6 = self.getattr_L__mod___features___4___se_act(getattr_l__mod___features___4___se_bn);  getattr_l__mod___features___4___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_7 = self.getattr_L__mod___features___4___se_fc2(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1 = x_se_7.sigmoid();  x_se_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_90 = x_89 * sigmoid_1;  x_89 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_91 = self.getattr_L__mod___features___4___act_dw(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_92 = self.getattr_L__mod___features___4___conv_pwl_conv(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___4___conv_pwl_bn_running_mean = self.getattr_L__mod___features___4___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___4___conv_pwl_bn_running_var = self.getattr_L__mod___features___4___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___4___conv_pwl_bn_weight = self.getattr_L__mod___features___4___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___4___conv_pwl_bn_bias = self.getattr_L__mod___features___4___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_93 = torch.nn.functional.batch_norm(x_92, getattr_l__mod___features___4___conv_pwl_bn_running_mean, getattr_l__mod___features___4___conv_pwl_bn_running_var, getattr_l__mod___features___4___conv_pwl_bn_weight, getattr_l__mod___features___4___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_92 = getattr_l__mod___features___4___conv_pwl_bn_running_mean = getattr_l__mod___features___4___conv_pwl_bn_running_var = getattr_l__mod___features___4___conv_pwl_bn_weight = getattr_l__mod___features___4___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_94 = self.getattr_L__mod___features___4___conv_pwl_bn_drop(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_97 = self.getattr_L__mod___features___4___conv_pwl_bn_act(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_2 = x_97[(slice(None, None, None), slice(0, 50, None))]
    add_1 = getitem_2 + shortcut_4;  getitem_2 = shortcut_4 = None
    getitem_3 = x_97[(slice(None, None, None), slice(50, None, None))];  x_97 = None
    shortcut_5 = torch.cat([add_1, getitem_3], dim = 1);  add_1 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_99 = self.getattr_L__mod___features___5___conv_exp_conv(shortcut_5);  shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___5___conv_exp_bn_running_mean = self.getattr_L__mod___features___5___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___5___conv_exp_bn_running_var = self.getattr_L__mod___features___5___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___5___conv_exp_bn_weight = self.getattr_L__mod___features___5___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___5___conv_exp_bn_bias = self.getattr_L__mod___features___5___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_100 = torch.nn.functional.batch_norm(x_99, getattr_l__mod___features___5___conv_exp_bn_running_mean, getattr_l__mod___features___5___conv_exp_bn_running_var, getattr_l__mod___features___5___conv_exp_bn_weight, getattr_l__mod___features___5___conv_exp_bn_bias, False, 0.1, 1e-05);  x_99 = getattr_l__mod___features___5___conv_exp_bn_running_mean = getattr_l__mod___features___5___conv_exp_bn_running_var = getattr_l__mod___features___5___conv_exp_bn_weight = getattr_l__mod___features___5___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_101 = self.getattr_L__mod___features___5___conv_exp_bn_drop(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_104 = self.getattr_L__mod___features___5___conv_exp_bn_act(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_105 = self.getattr_L__mod___features___5___conv_dw_conv(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___5___conv_dw_bn_running_mean = self.getattr_L__mod___features___5___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___5___conv_dw_bn_running_var = self.getattr_L__mod___features___5___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___5___conv_dw_bn_weight = self.getattr_L__mod___features___5___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___5___conv_dw_bn_bias = self.getattr_L__mod___features___5___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_106 = torch.nn.functional.batch_norm(x_105, getattr_l__mod___features___5___conv_dw_bn_running_mean, getattr_l__mod___features___5___conv_dw_bn_running_var, getattr_l__mod___features___5___conv_dw_bn_weight, getattr_l__mod___features___5___conv_dw_bn_bias, False, 0.1, 1e-05);  x_105 = getattr_l__mod___features___5___conv_dw_bn_running_mean = getattr_l__mod___features___5___conv_dw_bn_running_var = getattr_l__mod___features___5___conv_dw_bn_weight = getattr_l__mod___features___5___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_107 = self.getattr_L__mod___features___5___conv_dw_bn_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_110 = self.getattr_L__mod___features___5___conv_dw_bn_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_110.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_9 = self.getattr_L__mod___features___5___se_fc1(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___5___se_bn = self.getattr_L__mod___features___5___se_bn(x_se_9);  x_se_9 = None
    x_se_10 = self.getattr_L__mod___features___5___se_act(getattr_l__mod___features___5___se_bn);  getattr_l__mod___features___5___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_11 = self.getattr_L__mod___features___5___se_fc2(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2 = x_se_11.sigmoid();  x_se_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_111 = x_110 * sigmoid_2;  x_110 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_112 = self.getattr_L__mod___features___5___act_dw(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_113 = self.getattr_L__mod___features___5___conv_pwl_conv(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___5___conv_pwl_bn_running_mean = self.getattr_L__mod___features___5___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___5___conv_pwl_bn_running_var = self.getattr_L__mod___features___5___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___5___conv_pwl_bn_weight = self.getattr_L__mod___features___5___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___5___conv_pwl_bn_bias = self.getattr_L__mod___features___5___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_114 = torch.nn.functional.batch_norm(x_113, getattr_l__mod___features___5___conv_pwl_bn_running_mean, getattr_l__mod___features___5___conv_pwl_bn_running_var, getattr_l__mod___features___5___conv_pwl_bn_weight, getattr_l__mod___features___5___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_113 = getattr_l__mod___features___5___conv_pwl_bn_running_mean = getattr_l__mod___features___5___conv_pwl_bn_running_var = getattr_l__mod___features___5___conv_pwl_bn_weight = getattr_l__mod___features___5___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_115 = self.getattr_L__mod___features___5___conv_pwl_bn_drop(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_6 = self.getattr_L__mod___features___5___conv_pwl_bn_act(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_119 = self.getattr_L__mod___features___6___conv_exp_conv(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___6___conv_exp_bn_running_mean = self.getattr_L__mod___features___6___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___6___conv_exp_bn_running_var = self.getattr_L__mod___features___6___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___6___conv_exp_bn_weight = self.getattr_L__mod___features___6___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___6___conv_exp_bn_bias = self.getattr_L__mod___features___6___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_120 = torch.nn.functional.batch_norm(x_119, getattr_l__mod___features___6___conv_exp_bn_running_mean, getattr_l__mod___features___6___conv_exp_bn_running_var, getattr_l__mod___features___6___conv_exp_bn_weight, getattr_l__mod___features___6___conv_exp_bn_bias, False, 0.1, 1e-05);  x_119 = getattr_l__mod___features___6___conv_exp_bn_running_mean = getattr_l__mod___features___6___conv_exp_bn_running_var = getattr_l__mod___features___6___conv_exp_bn_weight = getattr_l__mod___features___6___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_121 = self.getattr_L__mod___features___6___conv_exp_bn_drop(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_124 = self.getattr_L__mod___features___6___conv_exp_bn_act(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_125 = self.getattr_L__mod___features___6___conv_dw_conv(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___6___conv_dw_bn_running_mean = self.getattr_L__mod___features___6___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___6___conv_dw_bn_running_var = self.getattr_L__mod___features___6___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___6___conv_dw_bn_weight = self.getattr_L__mod___features___6___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___6___conv_dw_bn_bias = self.getattr_L__mod___features___6___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_126 = torch.nn.functional.batch_norm(x_125, getattr_l__mod___features___6___conv_dw_bn_running_mean, getattr_l__mod___features___6___conv_dw_bn_running_var, getattr_l__mod___features___6___conv_dw_bn_weight, getattr_l__mod___features___6___conv_dw_bn_bias, False, 0.1, 1e-05);  x_125 = getattr_l__mod___features___6___conv_dw_bn_running_mean = getattr_l__mod___features___6___conv_dw_bn_running_var = getattr_l__mod___features___6___conv_dw_bn_weight = getattr_l__mod___features___6___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_127 = self.getattr_L__mod___features___6___conv_dw_bn_drop(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_130 = self.getattr_L__mod___features___6___conv_dw_bn_act(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_130.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_13 = self.getattr_L__mod___features___6___se_fc1(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___6___se_bn = self.getattr_L__mod___features___6___se_bn(x_se_13);  x_se_13 = None
    x_se_14 = self.getattr_L__mod___features___6___se_act(getattr_l__mod___features___6___se_bn);  getattr_l__mod___features___6___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_15 = self.getattr_L__mod___features___6___se_fc2(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3 = x_se_15.sigmoid();  x_se_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_131 = x_130 * sigmoid_3;  x_130 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_132 = self.getattr_L__mod___features___6___act_dw(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_133 = self.getattr_L__mod___features___6___conv_pwl_conv(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___6___conv_pwl_bn_running_mean = self.getattr_L__mod___features___6___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___6___conv_pwl_bn_running_var = self.getattr_L__mod___features___6___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___6___conv_pwl_bn_weight = self.getattr_L__mod___features___6___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___6___conv_pwl_bn_bias = self.getattr_L__mod___features___6___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_134 = torch.nn.functional.batch_norm(x_133, getattr_l__mod___features___6___conv_pwl_bn_running_mean, getattr_l__mod___features___6___conv_pwl_bn_running_var, getattr_l__mod___features___6___conv_pwl_bn_weight, getattr_l__mod___features___6___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_133 = getattr_l__mod___features___6___conv_pwl_bn_running_mean = getattr_l__mod___features___6___conv_pwl_bn_running_var = getattr_l__mod___features___6___conv_pwl_bn_weight = getattr_l__mod___features___6___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_135 = self.getattr_L__mod___features___6___conv_pwl_bn_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_138 = self.getattr_L__mod___features___6___conv_pwl_bn_act(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_4 = x_138[(slice(None, None, None), slice(0, 72, None))]
    add_2 = getitem_4 + shortcut_6;  getitem_4 = shortcut_6 = None
    getitem_5 = x_138[(slice(None, None, None), slice(72, None, None))];  x_138 = None
    shortcut_7 = torch.cat([add_2, getitem_5], dim = 1);  add_2 = getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_140 = self.getattr_L__mod___features___7___conv_exp_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___7___conv_exp_bn_running_mean = self.getattr_L__mod___features___7___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___7___conv_exp_bn_running_var = self.getattr_L__mod___features___7___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___7___conv_exp_bn_weight = self.getattr_L__mod___features___7___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___7___conv_exp_bn_bias = self.getattr_L__mod___features___7___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_141 = torch.nn.functional.batch_norm(x_140, getattr_l__mod___features___7___conv_exp_bn_running_mean, getattr_l__mod___features___7___conv_exp_bn_running_var, getattr_l__mod___features___7___conv_exp_bn_weight, getattr_l__mod___features___7___conv_exp_bn_bias, False, 0.1, 1e-05);  x_140 = getattr_l__mod___features___7___conv_exp_bn_running_mean = getattr_l__mod___features___7___conv_exp_bn_running_var = getattr_l__mod___features___7___conv_exp_bn_weight = getattr_l__mod___features___7___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_142 = self.getattr_L__mod___features___7___conv_exp_bn_drop(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_145 = self.getattr_L__mod___features___7___conv_exp_bn_act(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_146 = self.getattr_L__mod___features___7___conv_dw_conv(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___7___conv_dw_bn_running_mean = self.getattr_L__mod___features___7___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___7___conv_dw_bn_running_var = self.getattr_L__mod___features___7___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___7___conv_dw_bn_weight = self.getattr_L__mod___features___7___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___7___conv_dw_bn_bias = self.getattr_L__mod___features___7___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_147 = torch.nn.functional.batch_norm(x_146, getattr_l__mod___features___7___conv_dw_bn_running_mean, getattr_l__mod___features___7___conv_dw_bn_running_var, getattr_l__mod___features___7___conv_dw_bn_weight, getattr_l__mod___features___7___conv_dw_bn_bias, False, 0.1, 1e-05);  x_146 = getattr_l__mod___features___7___conv_dw_bn_running_mean = getattr_l__mod___features___7___conv_dw_bn_running_var = getattr_l__mod___features___7___conv_dw_bn_weight = getattr_l__mod___features___7___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_148 = self.getattr_L__mod___features___7___conv_dw_bn_drop(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_151 = self.getattr_L__mod___features___7___conv_dw_bn_act(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_151.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_17 = self.getattr_L__mod___features___7___se_fc1(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___7___se_bn = self.getattr_L__mod___features___7___se_bn(x_se_17);  x_se_17 = None
    x_se_18 = self.getattr_L__mod___features___7___se_act(getattr_l__mod___features___7___se_bn);  getattr_l__mod___features___7___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_19 = self.getattr_L__mod___features___7___se_fc2(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4 = x_se_19.sigmoid();  x_se_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_152 = x_151 * sigmoid_4;  x_151 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_153 = self.getattr_L__mod___features___7___act_dw(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_154 = self.getattr_L__mod___features___7___conv_pwl_conv(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___7___conv_pwl_bn_running_mean = self.getattr_L__mod___features___7___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___7___conv_pwl_bn_running_var = self.getattr_L__mod___features___7___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___7___conv_pwl_bn_weight = self.getattr_L__mod___features___7___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___7___conv_pwl_bn_bias = self.getattr_L__mod___features___7___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_155 = torch.nn.functional.batch_norm(x_154, getattr_l__mod___features___7___conv_pwl_bn_running_mean, getattr_l__mod___features___7___conv_pwl_bn_running_var, getattr_l__mod___features___7___conv_pwl_bn_weight, getattr_l__mod___features___7___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_154 = getattr_l__mod___features___7___conv_pwl_bn_running_mean = getattr_l__mod___features___7___conv_pwl_bn_running_var = getattr_l__mod___features___7___conv_pwl_bn_weight = getattr_l__mod___features___7___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_156 = self.getattr_L__mod___features___7___conv_pwl_bn_drop(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_159 = self.getattr_L__mod___features___7___conv_pwl_bn_act(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_6 = x_159[(slice(None, None, None), slice(0, 84, None))]
    add_3 = getitem_6 + shortcut_7;  getitem_6 = shortcut_7 = None
    getitem_7 = x_159[(slice(None, None, None), slice(84, None, None))];  x_159 = None
    shortcut_8 = torch.cat([add_3, getitem_7], dim = 1);  add_3 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_161 = self.getattr_L__mod___features___8___conv_exp_conv(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___8___conv_exp_bn_running_mean = self.getattr_L__mod___features___8___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___8___conv_exp_bn_running_var = self.getattr_L__mod___features___8___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___8___conv_exp_bn_weight = self.getattr_L__mod___features___8___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___8___conv_exp_bn_bias = self.getattr_L__mod___features___8___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_162 = torch.nn.functional.batch_norm(x_161, getattr_l__mod___features___8___conv_exp_bn_running_mean, getattr_l__mod___features___8___conv_exp_bn_running_var, getattr_l__mod___features___8___conv_exp_bn_weight, getattr_l__mod___features___8___conv_exp_bn_bias, False, 0.1, 1e-05);  x_161 = getattr_l__mod___features___8___conv_exp_bn_running_mean = getattr_l__mod___features___8___conv_exp_bn_running_var = getattr_l__mod___features___8___conv_exp_bn_weight = getattr_l__mod___features___8___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_163 = self.getattr_L__mod___features___8___conv_exp_bn_drop(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_166 = self.getattr_L__mod___features___8___conv_exp_bn_act(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_167 = self.getattr_L__mod___features___8___conv_dw_conv(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___8___conv_dw_bn_running_mean = self.getattr_L__mod___features___8___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___8___conv_dw_bn_running_var = self.getattr_L__mod___features___8___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___8___conv_dw_bn_weight = self.getattr_L__mod___features___8___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___8___conv_dw_bn_bias = self.getattr_L__mod___features___8___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_168 = torch.nn.functional.batch_norm(x_167, getattr_l__mod___features___8___conv_dw_bn_running_mean, getattr_l__mod___features___8___conv_dw_bn_running_var, getattr_l__mod___features___8___conv_dw_bn_weight, getattr_l__mod___features___8___conv_dw_bn_bias, False, 0.1, 1e-05);  x_167 = getattr_l__mod___features___8___conv_dw_bn_running_mean = getattr_l__mod___features___8___conv_dw_bn_running_var = getattr_l__mod___features___8___conv_dw_bn_weight = getattr_l__mod___features___8___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_169 = self.getattr_L__mod___features___8___conv_dw_bn_drop(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_172 = self.getattr_L__mod___features___8___conv_dw_bn_act(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_172.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_21 = self.getattr_L__mod___features___8___se_fc1(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___8___se_bn = self.getattr_L__mod___features___8___se_bn(x_se_21);  x_se_21 = None
    x_se_22 = self.getattr_L__mod___features___8___se_act(getattr_l__mod___features___8___se_bn);  getattr_l__mod___features___8___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_23 = self.getattr_L__mod___features___8___se_fc2(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5 = x_se_23.sigmoid();  x_se_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_173 = x_172 * sigmoid_5;  x_172 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_174 = self.getattr_L__mod___features___8___act_dw(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_175 = self.getattr_L__mod___features___8___conv_pwl_conv(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___8___conv_pwl_bn_running_mean = self.getattr_L__mod___features___8___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___8___conv_pwl_bn_running_var = self.getattr_L__mod___features___8___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___8___conv_pwl_bn_weight = self.getattr_L__mod___features___8___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___8___conv_pwl_bn_bias = self.getattr_L__mod___features___8___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_176 = torch.nn.functional.batch_norm(x_175, getattr_l__mod___features___8___conv_pwl_bn_running_mean, getattr_l__mod___features___8___conv_pwl_bn_running_var, getattr_l__mod___features___8___conv_pwl_bn_weight, getattr_l__mod___features___8___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_175 = getattr_l__mod___features___8___conv_pwl_bn_running_mean = getattr_l__mod___features___8___conv_pwl_bn_running_var = getattr_l__mod___features___8___conv_pwl_bn_weight = getattr_l__mod___features___8___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_177 = self.getattr_L__mod___features___8___conv_pwl_bn_drop(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_180 = self.getattr_L__mod___features___8___conv_pwl_bn_act(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_8 = x_180[(slice(None, None, None), slice(0, 95, None))]
    add_4 = getitem_8 + shortcut_8;  getitem_8 = shortcut_8 = None
    getitem_9 = x_180[(slice(None, None, None), slice(95, None, None))];  x_180 = None
    shortcut_9 = torch.cat([add_4, getitem_9], dim = 1);  add_4 = getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_182 = self.getattr_L__mod___features___9___conv_exp_conv(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___9___conv_exp_bn_running_mean = self.getattr_L__mod___features___9___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___9___conv_exp_bn_running_var = self.getattr_L__mod___features___9___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___9___conv_exp_bn_weight = self.getattr_L__mod___features___9___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___9___conv_exp_bn_bias = self.getattr_L__mod___features___9___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_183 = torch.nn.functional.batch_norm(x_182, getattr_l__mod___features___9___conv_exp_bn_running_mean, getattr_l__mod___features___9___conv_exp_bn_running_var, getattr_l__mod___features___9___conv_exp_bn_weight, getattr_l__mod___features___9___conv_exp_bn_bias, False, 0.1, 1e-05);  x_182 = getattr_l__mod___features___9___conv_exp_bn_running_mean = getattr_l__mod___features___9___conv_exp_bn_running_var = getattr_l__mod___features___9___conv_exp_bn_weight = getattr_l__mod___features___9___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_184 = self.getattr_L__mod___features___9___conv_exp_bn_drop(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_187 = self.getattr_L__mod___features___9___conv_exp_bn_act(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_188 = self.getattr_L__mod___features___9___conv_dw_conv(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___9___conv_dw_bn_running_mean = self.getattr_L__mod___features___9___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___9___conv_dw_bn_running_var = self.getattr_L__mod___features___9___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___9___conv_dw_bn_weight = self.getattr_L__mod___features___9___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___9___conv_dw_bn_bias = self.getattr_L__mod___features___9___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_189 = torch.nn.functional.batch_norm(x_188, getattr_l__mod___features___9___conv_dw_bn_running_mean, getattr_l__mod___features___9___conv_dw_bn_running_var, getattr_l__mod___features___9___conv_dw_bn_weight, getattr_l__mod___features___9___conv_dw_bn_bias, False, 0.1, 1e-05);  x_188 = getattr_l__mod___features___9___conv_dw_bn_running_mean = getattr_l__mod___features___9___conv_dw_bn_running_var = getattr_l__mod___features___9___conv_dw_bn_weight = getattr_l__mod___features___9___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_190 = self.getattr_L__mod___features___9___conv_dw_bn_drop(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_193 = self.getattr_L__mod___features___9___conv_dw_bn_act(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_193.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_25 = self.getattr_L__mod___features___9___se_fc1(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___9___se_bn = self.getattr_L__mod___features___9___se_bn(x_se_25);  x_se_25 = None
    x_se_26 = self.getattr_L__mod___features___9___se_act(getattr_l__mod___features___9___se_bn);  getattr_l__mod___features___9___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_27 = self.getattr_L__mod___features___9___se_fc2(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6 = x_se_27.sigmoid();  x_se_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_194 = x_193 * sigmoid_6;  x_193 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_195 = self.getattr_L__mod___features___9___act_dw(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_196 = self.getattr_L__mod___features___9___conv_pwl_conv(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___9___conv_pwl_bn_running_mean = self.getattr_L__mod___features___9___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___9___conv_pwl_bn_running_var = self.getattr_L__mod___features___9___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___9___conv_pwl_bn_weight = self.getattr_L__mod___features___9___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___9___conv_pwl_bn_bias = self.getattr_L__mod___features___9___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_197 = torch.nn.functional.batch_norm(x_196, getattr_l__mod___features___9___conv_pwl_bn_running_mean, getattr_l__mod___features___9___conv_pwl_bn_running_var, getattr_l__mod___features___9___conv_pwl_bn_weight, getattr_l__mod___features___9___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_196 = getattr_l__mod___features___9___conv_pwl_bn_running_mean = getattr_l__mod___features___9___conv_pwl_bn_running_var = getattr_l__mod___features___9___conv_pwl_bn_weight = getattr_l__mod___features___9___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_198 = self.getattr_L__mod___features___9___conv_pwl_bn_drop(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_201 = self.getattr_L__mod___features___9___conv_pwl_bn_act(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_10 = x_201[(slice(None, None, None), slice(0, 106, None))]
    add_5 = getitem_10 + shortcut_9;  getitem_10 = shortcut_9 = None
    getitem_11 = x_201[(slice(None, None, None), slice(106, None, None))];  x_201 = None
    shortcut_10 = torch.cat([add_5, getitem_11], dim = 1);  add_5 = getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_203 = self.getattr_L__mod___features___10___conv_exp_conv(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___10___conv_exp_bn_running_mean = self.getattr_L__mod___features___10___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___10___conv_exp_bn_running_var = self.getattr_L__mod___features___10___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___10___conv_exp_bn_weight = self.getattr_L__mod___features___10___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___10___conv_exp_bn_bias = self.getattr_L__mod___features___10___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_204 = torch.nn.functional.batch_norm(x_203, getattr_l__mod___features___10___conv_exp_bn_running_mean, getattr_l__mod___features___10___conv_exp_bn_running_var, getattr_l__mod___features___10___conv_exp_bn_weight, getattr_l__mod___features___10___conv_exp_bn_bias, False, 0.1, 1e-05);  x_203 = getattr_l__mod___features___10___conv_exp_bn_running_mean = getattr_l__mod___features___10___conv_exp_bn_running_var = getattr_l__mod___features___10___conv_exp_bn_weight = getattr_l__mod___features___10___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_205 = self.getattr_L__mod___features___10___conv_exp_bn_drop(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_208 = self.getattr_L__mod___features___10___conv_exp_bn_act(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_209 = self.getattr_L__mod___features___10___conv_dw_conv(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___10___conv_dw_bn_running_mean = self.getattr_L__mod___features___10___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___10___conv_dw_bn_running_var = self.getattr_L__mod___features___10___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___10___conv_dw_bn_weight = self.getattr_L__mod___features___10___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___10___conv_dw_bn_bias = self.getattr_L__mod___features___10___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_210 = torch.nn.functional.batch_norm(x_209, getattr_l__mod___features___10___conv_dw_bn_running_mean, getattr_l__mod___features___10___conv_dw_bn_running_var, getattr_l__mod___features___10___conv_dw_bn_weight, getattr_l__mod___features___10___conv_dw_bn_bias, False, 0.1, 1e-05);  x_209 = getattr_l__mod___features___10___conv_dw_bn_running_mean = getattr_l__mod___features___10___conv_dw_bn_running_var = getattr_l__mod___features___10___conv_dw_bn_weight = getattr_l__mod___features___10___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_211 = self.getattr_L__mod___features___10___conv_dw_bn_drop(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_214 = self.getattr_L__mod___features___10___conv_dw_bn_act(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_214.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_29 = self.getattr_L__mod___features___10___se_fc1(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___10___se_bn = self.getattr_L__mod___features___10___se_bn(x_se_29);  x_se_29 = None
    x_se_30 = self.getattr_L__mod___features___10___se_act(getattr_l__mod___features___10___se_bn);  getattr_l__mod___features___10___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_31 = self.getattr_L__mod___features___10___se_fc2(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7 = x_se_31.sigmoid();  x_se_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_215 = x_214 * sigmoid_7;  x_214 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_216 = self.getattr_L__mod___features___10___act_dw(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_217 = self.getattr_L__mod___features___10___conv_pwl_conv(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___10___conv_pwl_bn_running_mean = self.getattr_L__mod___features___10___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___10___conv_pwl_bn_running_var = self.getattr_L__mod___features___10___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___10___conv_pwl_bn_weight = self.getattr_L__mod___features___10___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___10___conv_pwl_bn_bias = self.getattr_L__mod___features___10___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_218 = torch.nn.functional.batch_norm(x_217, getattr_l__mod___features___10___conv_pwl_bn_running_mean, getattr_l__mod___features___10___conv_pwl_bn_running_var, getattr_l__mod___features___10___conv_pwl_bn_weight, getattr_l__mod___features___10___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_217 = getattr_l__mod___features___10___conv_pwl_bn_running_mean = getattr_l__mod___features___10___conv_pwl_bn_running_var = getattr_l__mod___features___10___conv_pwl_bn_weight = getattr_l__mod___features___10___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_219 = self.getattr_L__mod___features___10___conv_pwl_bn_drop(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_222 = self.getattr_L__mod___features___10___conv_pwl_bn_act(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_12 = x_222[(slice(None, None, None), slice(0, 117, None))]
    add_6 = getitem_12 + shortcut_10;  getitem_12 = shortcut_10 = None
    getitem_13 = x_222[(slice(None, None, None), slice(117, None, None))];  x_222 = None
    shortcut_11 = torch.cat([add_6, getitem_13], dim = 1);  add_6 = getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_224 = self.getattr_L__mod___features___11___conv_exp_conv(shortcut_11);  shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___11___conv_exp_bn_running_mean = self.getattr_L__mod___features___11___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___11___conv_exp_bn_running_var = self.getattr_L__mod___features___11___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___11___conv_exp_bn_weight = self.getattr_L__mod___features___11___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___11___conv_exp_bn_bias = self.getattr_L__mod___features___11___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_225 = torch.nn.functional.batch_norm(x_224, getattr_l__mod___features___11___conv_exp_bn_running_mean, getattr_l__mod___features___11___conv_exp_bn_running_var, getattr_l__mod___features___11___conv_exp_bn_weight, getattr_l__mod___features___11___conv_exp_bn_bias, False, 0.1, 1e-05);  x_224 = getattr_l__mod___features___11___conv_exp_bn_running_mean = getattr_l__mod___features___11___conv_exp_bn_running_var = getattr_l__mod___features___11___conv_exp_bn_weight = getattr_l__mod___features___11___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_226 = self.getattr_L__mod___features___11___conv_exp_bn_drop(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_229 = self.getattr_L__mod___features___11___conv_exp_bn_act(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_230 = self.getattr_L__mod___features___11___conv_dw_conv(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___11___conv_dw_bn_running_mean = self.getattr_L__mod___features___11___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___11___conv_dw_bn_running_var = self.getattr_L__mod___features___11___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___11___conv_dw_bn_weight = self.getattr_L__mod___features___11___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___11___conv_dw_bn_bias = self.getattr_L__mod___features___11___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_231 = torch.nn.functional.batch_norm(x_230, getattr_l__mod___features___11___conv_dw_bn_running_mean, getattr_l__mod___features___11___conv_dw_bn_running_var, getattr_l__mod___features___11___conv_dw_bn_weight, getattr_l__mod___features___11___conv_dw_bn_bias, False, 0.1, 1e-05);  x_230 = getattr_l__mod___features___11___conv_dw_bn_running_mean = getattr_l__mod___features___11___conv_dw_bn_running_var = getattr_l__mod___features___11___conv_dw_bn_weight = getattr_l__mod___features___11___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_232 = self.getattr_L__mod___features___11___conv_dw_bn_drop(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_235 = self.getattr_L__mod___features___11___conv_dw_bn_act(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = x_235.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_33 = self.getattr_L__mod___features___11___se_fc1(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___11___se_bn = self.getattr_L__mod___features___11___se_bn(x_se_33);  x_se_33 = None
    x_se_34 = self.getattr_L__mod___features___11___se_act(getattr_l__mod___features___11___se_bn);  getattr_l__mod___features___11___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_35 = self.getattr_L__mod___features___11___se_fc2(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8 = x_se_35.sigmoid();  x_se_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_236 = x_235 * sigmoid_8;  x_235 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_237 = self.getattr_L__mod___features___11___act_dw(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_238 = self.getattr_L__mod___features___11___conv_pwl_conv(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___11___conv_pwl_bn_running_mean = self.getattr_L__mod___features___11___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___11___conv_pwl_bn_running_var = self.getattr_L__mod___features___11___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___11___conv_pwl_bn_weight = self.getattr_L__mod___features___11___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___11___conv_pwl_bn_bias = self.getattr_L__mod___features___11___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_239 = torch.nn.functional.batch_norm(x_238, getattr_l__mod___features___11___conv_pwl_bn_running_mean, getattr_l__mod___features___11___conv_pwl_bn_running_var, getattr_l__mod___features___11___conv_pwl_bn_weight, getattr_l__mod___features___11___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_238 = getattr_l__mod___features___11___conv_pwl_bn_running_mean = getattr_l__mod___features___11___conv_pwl_bn_running_var = getattr_l__mod___features___11___conv_pwl_bn_weight = getattr_l__mod___features___11___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_240 = self.getattr_L__mod___features___11___conv_pwl_bn_drop(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_12 = self.getattr_L__mod___features___11___conv_pwl_bn_act(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_244 = self.getattr_L__mod___features___12___conv_exp_conv(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___12___conv_exp_bn_running_mean = self.getattr_L__mod___features___12___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___12___conv_exp_bn_running_var = self.getattr_L__mod___features___12___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___12___conv_exp_bn_weight = self.getattr_L__mod___features___12___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___12___conv_exp_bn_bias = self.getattr_L__mod___features___12___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_245 = torch.nn.functional.batch_norm(x_244, getattr_l__mod___features___12___conv_exp_bn_running_mean, getattr_l__mod___features___12___conv_exp_bn_running_var, getattr_l__mod___features___12___conv_exp_bn_weight, getattr_l__mod___features___12___conv_exp_bn_bias, False, 0.1, 1e-05);  x_244 = getattr_l__mod___features___12___conv_exp_bn_running_mean = getattr_l__mod___features___12___conv_exp_bn_running_var = getattr_l__mod___features___12___conv_exp_bn_weight = getattr_l__mod___features___12___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_246 = self.getattr_L__mod___features___12___conv_exp_bn_drop(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_249 = self.getattr_L__mod___features___12___conv_exp_bn_act(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_250 = self.getattr_L__mod___features___12___conv_dw_conv(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___12___conv_dw_bn_running_mean = self.getattr_L__mod___features___12___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___12___conv_dw_bn_running_var = self.getattr_L__mod___features___12___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___12___conv_dw_bn_weight = self.getattr_L__mod___features___12___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___12___conv_dw_bn_bias = self.getattr_L__mod___features___12___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_251 = torch.nn.functional.batch_norm(x_250, getattr_l__mod___features___12___conv_dw_bn_running_mean, getattr_l__mod___features___12___conv_dw_bn_running_var, getattr_l__mod___features___12___conv_dw_bn_weight, getattr_l__mod___features___12___conv_dw_bn_bias, False, 0.1, 1e-05);  x_250 = getattr_l__mod___features___12___conv_dw_bn_running_mean = getattr_l__mod___features___12___conv_dw_bn_running_var = getattr_l__mod___features___12___conv_dw_bn_weight = getattr_l__mod___features___12___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_252 = self.getattr_L__mod___features___12___conv_dw_bn_drop(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_255 = self.getattr_L__mod___features___12___conv_dw_bn_act(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = x_255.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_37 = self.getattr_L__mod___features___12___se_fc1(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___12___se_bn = self.getattr_L__mod___features___12___se_bn(x_se_37);  x_se_37 = None
    x_se_38 = self.getattr_L__mod___features___12___se_act(getattr_l__mod___features___12___se_bn);  getattr_l__mod___features___12___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_39 = self.getattr_L__mod___features___12___se_fc2(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9 = x_se_39.sigmoid();  x_se_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_256 = x_255 * sigmoid_9;  x_255 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_257 = self.getattr_L__mod___features___12___act_dw(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_258 = self.getattr_L__mod___features___12___conv_pwl_conv(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___12___conv_pwl_bn_running_mean = self.getattr_L__mod___features___12___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___12___conv_pwl_bn_running_var = self.getattr_L__mod___features___12___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___12___conv_pwl_bn_weight = self.getattr_L__mod___features___12___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___12___conv_pwl_bn_bias = self.getattr_L__mod___features___12___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_259 = torch.nn.functional.batch_norm(x_258, getattr_l__mod___features___12___conv_pwl_bn_running_mean, getattr_l__mod___features___12___conv_pwl_bn_running_var, getattr_l__mod___features___12___conv_pwl_bn_weight, getattr_l__mod___features___12___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_258 = getattr_l__mod___features___12___conv_pwl_bn_running_mean = getattr_l__mod___features___12___conv_pwl_bn_running_var = getattr_l__mod___features___12___conv_pwl_bn_weight = getattr_l__mod___features___12___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_260 = self.getattr_L__mod___features___12___conv_pwl_bn_drop(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_263 = self.getattr_L__mod___features___12___conv_pwl_bn_act(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_14 = x_263[(slice(None, None, None), slice(0, 140, None))]
    add_7 = getitem_14 + shortcut_12;  getitem_14 = shortcut_12 = None
    getitem_15 = x_263[(slice(None, None, None), slice(140, None, None))];  x_263 = None
    shortcut_13 = torch.cat([add_7, getitem_15], dim = 1);  add_7 = getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_265 = self.getattr_L__mod___features___13___conv_exp_conv(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___13___conv_exp_bn_running_mean = self.getattr_L__mod___features___13___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___13___conv_exp_bn_running_var = self.getattr_L__mod___features___13___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___13___conv_exp_bn_weight = self.getattr_L__mod___features___13___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___13___conv_exp_bn_bias = self.getattr_L__mod___features___13___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_266 = torch.nn.functional.batch_norm(x_265, getattr_l__mod___features___13___conv_exp_bn_running_mean, getattr_l__mod___features___13___conv_exp_bn_running_var, getattr_l__mod___features___13___conv_exp_bn_weight, getattr_l__mod___features___13___conv_exp_bn_bias, False, 0.1, 1e-05);  x_265 = getattr_l__mod___features___13___conv_exp_bn_running_mean = getattr_l__mod___features___13___conv_exp_bn_running_var = getattr_l__mod___features___13___conv_exp_bn_weight = getattr_l__mod___features___13___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_267 = self.getattr_L__mod___features___13___conv_exp_bn_drop(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_270 = self.getattr_L__mod___features___13___conv_exp_bn_act(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_271 = self.getattr_L__mod___features___13___conv_dw_conv(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___13___conv_dw_bn_running_mean = self.getattr_L__mod___features___13___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___13___conv_dw_bn_running_var = self.getattr_L__mod___features___13___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___13___conv_dw_bn_weight = self.getattr_L__mod___features___13___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___13___conv_dw_bn_bias = self.getattr_L__mod___features___13___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_272 = torch.nn.functional.batch_norm(x_271, getattr_l__mod___features___13___conv_dw_bn_running_mean, getattr_l__mod___features___13___conv_dw_bn_running_var, getattr_l__mod___features___13___conv_dw_bn_weight, getattr_l__mod___features___13___conv_dw_bn_bias, False, 0.1, 1e-05);  x_271 = getattr_l__mod___features___13___conv_dw_bn_running_mean = getattr_l__mod___features___13___conv_dw_bn_running_var = getattr_l__mod___features___13___conv_dw_bn_weight = getattr_l__mod___features___13___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_273 = self.getattr_L__mod___features___13___conv_dw_bn_drop(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_276 = self.getattr_L__mod___features___13___conv_dw_bn_act(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = x_276.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_41 = self.getattr_L__mod___features___13___se_fc1(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___13___se_bn = self.getattr_L__mod___features___13___se_bn(x_se_41);  x_se_41 = None
    x_se_42 = self.getattr_L__mod___features___13___se_act(getattr_l__mod___features___13___se_bn);  getattr_l__mod___features___13___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_43 = self.getattr_L__mod___features___13___se_fc2(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10 = x_se_43.sigmoid();  x_se_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_277 = x_276 * sigmoid_10;  x_276 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_278 = self.getattr_L__mod___features___13___act_dw(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_279 = self.getattr_L__mod___features___13___conv_pwl_conv(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___13___conv_pwl_bn_running_mean = self.getattr_L__mod___features___13___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___13___conv_pwl_bn_running_var = self.getattr_L__mod___features___13___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___13___conv_pwl_bn_weight = self.getattr_L__mod___features___13___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___13___conv_pwl_bn_bias = self.getattr_L__mod___features___13___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_280 = torch.nn.functional.batch_norm(x_279, getattr_l__mod___features___13___conv_pwl_bn_running_mean, getattr_l__mod___features___13___conv_pwl_bn_running_var, getattr_l__mod___features___13___conv_pwl_bn_weight, getattr_l__mod___features___13___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_279 = getattr_l__mod___features___13___conv_pwl_bn_running_mean = getattr_l__mod___features___13___conv_pwl_bn_running_var = getattr_l__mod___features___13___conv_pwl_bn_weight = getattr_l__mod___features___13___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_281 = self.getattr_L__mod___features___13___conv_pwl_bn_drop(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_284 = self.getattr_L__mod___features___13___conv_pwl_bn_act(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_16 = x_284[(slice(None, None, None), slice(0, 151, None))]
    add_8 = getitem_16 + shortcut_13;  getitem_16 = shortcut_13 = None
    getitem_17 = x_284[(slice(None, None, None), slice(151, None, None))];  x_284 = None
    shortcut_14 = torch.cat([add_8, getitem_17], dim = 1);  add_8 = getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_286 = self.getattr_L__mod___features___14___conv_exp_conv(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___14___conv_exp_bn_running_mean = self.getattr_L__mod___features___14___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___14___conv_exp_bn_running_var = self.getattr_L__mod___features___14___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___14___conv_exp_bn_weight = self.getattr_L__mod___features___14___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___14___conv_exp_bn_bias = self.getattr_L__mod___features___14___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_287 = torch.nn.functional.batch_norm(x_286, getattr_l__mod___features___14___conv_exp_bn_running_mean, getattr_l__mod___features___14___conv_exp_bn_running_var, getattr_l__mod___features___14___conv_exp_bn_weight, getattr_l__mod___features___14___conv_exp_bn_bias, False, 0.1, 1e-05);  x_286 = getattr_l__mod___features___14___conv_exp_bn_running_mean = getattr_l__mod___features___14___conv_exp_bn_running_var = getattr_l__mod___features___14___conv_exp_bn_weight = getattr_l__mod___features___14___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_288 = self.getattr_L__mod___features___14___conv_exp_bn_drop(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_291 = self.getattr_L__mod___features___14___conv_exp_bn_act(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_292 = self.getattr_L__mod___features___14___conv_dw_conv(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___14___conv_dw_bn_running_mean = self.getattr_L__mod___features___14___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___14___conv_dw_bn_running_var = self.getattr_L__mod___features___14___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___14___conv_dw_bn_weight = self.getattr_L__mod___features___14___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___14___conv_dw_bn_bias = self.getattr_L__mod___features___14___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_293 = torch.nn.functional.batch_norm(x_292, getattr_l__mod___features___14___conv_dw_bn_running_mean, getattr_l__mod___features___14___conv_dw_bn_running_var, getattr_l__mod___features___14___conv_dw_bn_weight, getattr_l__mod___features___14___conv_dw_bn_bias, False, 0.1, 1e-05);  x_292 = getattr_l__mod___features___14___conv_dw_bn_running_mean = getattr_l__mod___features___14___conv_dw_bn_running_var = getattr_l__mod___features___14___conv_dw_bn_weight = getattr_l__mod___features___14___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_294 = self.getattr_L__mod___features___14___conv_dw_bn_drop(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_297 = self.getattr_L__mod___features___14___conv_dw_bn_act(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = x_297.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_45 = self.getattr_L__mod___features___14___se_fc1(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___14___se_bn = self.getattr_L__mod___features___14___se_bn(x_se_45);  x_se_45 = None
    x_se_46 = self.getattr_L__mod___features___14___se_act(getattr_l__mod___features___14___se_bn);  getattr_l__mod___features___14___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_47 = self.getattr_L__mod___features___14___se_fc2(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11 = x_se_47.sigmoid();  x_se_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_298 = x_297 * sigmoid_11;  x_297 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_299 = self.getattr_L__mod___features___14___act_dw(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_300 = self.getattr_L__mod___features___14___conv_pwl_conv(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___14___conv_pwl_bn_running_mean = self.getattr_L__mod___features___14___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___14___conv_pwl_bn_running_var = self.getattr_L__mod___features___14___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___14___conv_pwl_bn_weight = self.getattr_L__mod___features___14___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___14___conv_pwl_bn_bias = self.getattr_L__mod___features___14___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_301 = torch.nn.functional.batch_norm(x_300, getattr_l__mod___features___14___conv_pwl_bn_running_mean, getattr_l__mod___features___14___conv_pwl_bn_running_var, getattr_l__mod___features___14___conv_pwl_bn_weight, getattr_l__mod___features___14___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_300 = getattr_l__mod___features___14___conv_pwl_bn_running_mean = getattr_l__mod___features___14___conv_pwl_bn_running_var = getattr_l__mod___features___14___conv_pwl_bn_weight = getattr_l__mod___features___14___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_302 = self.getattr_L__mod___features___14___conv_pwl_bn_drop(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_305 = self.getattr_L__mod___features___14___conv_pwl_bn_act(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_18 = x_305[(slice(None, None, None), slice(0, 162, None))]
    add_9 = getitem_18 + shortcut_14;  getitem_18 = shortcut_14 = None
    getitem_19 = x_305[(slice(None, None, None), slice(162, None, None))];  x_305 = None
    shortcut_15 = torch.cat([add_9, getitem_19], dim = 1);  add_9 = getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_307 = self.getattr_L__mod___features___15___conv_exp_conv(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___15___conv_exp_bn_running_mean = self.getattr_L__mod___features___15___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___15___conv_exp_bn_running_var = self.getattr_L__mod___features___15___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___15___conv_exp_bn_weight = self.getattr_L__mod___features___15___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___15___conv_exp_bn_bias = self.getattr_L__mod___features___15___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_308 = torch.nn.functional.batch_norm(x_307, getattr_l__mod___features___15___conv_exp_bn_running_mean, getattr_l__mod___features___15___conv_exp_bn_running_var, getattr_l__mod___features___15___conv_exp_bn_weight, getattr_l__mod___features___15___conv_exp_bn_bias, False, 0.1, 1e-05);  x_307 = getattr_l__mod___features___15___conv_exp_bn_running_mean = getattr_l__mod___features___15___conv_exp_bn_running_var = getattr_l__mod___features___15___conv_exp_bn_weight = getattr_l__mod___features___15___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_309 = self.getattr_L__mod___features___15___conv_exp_bn_drop(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_312 = self.getattr_L__mod___features___15___conv_exp_bn_act(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_313 = self.getattr_L__mod___features___15___conv_dw_conv(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___15___conv_dw_bn_running_mean = self.getattr_L__mod___features___15___conv_dw_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___15___conv_dw_bn_running_var = self.getattr_L__mod___features___15___conv_dw_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___15___conv_dw_bn_weight = self.getattr_L__mod___features___15___conv_dw_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___15___conv_dw_bn_bias = self.getattr_L__mod___features___15___conv_dw_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_314 = torch.nn.functional.batch_norm(x_313, getattr_l__mod___features___15___conv_dw_bn_running_mean, getattr_l__mod___features___15___conv_dw_bn_running_var, getattr_l__mod___features___15___conv_dw_bn_weight, getattr_l__mod___features___15___conv_dw_bn_bias, False, 0.1, 1e-05);  x_313 = getattr_l__mod___features___15___conv_dw_bn_running_mean = getattr_l__mod___features___15___conv_dw_bn_running_var = getattr_l__mod___features___15___conv_dw_bn_weight = getattr_l__mod___features___15___conv_dw_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_315 = self.getattr_L__mod___features___15___conv_dw_bn_drop(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_318 = self.getattr_L__mod___features___15___conv_dw_bn_act(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_48 = x_318.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_49 = self.getattr_L__mod___features___15___se_fc1(x_se_48);  x_se_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_l__mod___features___15___se_bn = self.getattr_L__mod___features___15___se_bn(x_se_49);  x_se_49 = None
    x_se_50 = self.getattr_L__mod___features___15___se_act(getattr_l__mod___features___15___se_bn);  getattr_l__mod___features___15___se_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_51 = self.getattr_L__mod___features___15___se_fc2(x_se_50);  x_se_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12 = x_se_51.sigmoid();  x_se_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_319 = x_318 * sigmoid_12;  x_318 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    x_320 = self.getattr_L__mod___features___15___act_dw(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_321 = self.getattr_L__mod___features___15___conv_pwl_conv(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___15___conv_pwl_bn_running_mean = self.getattr_L__mod___features___15___conv_pwl_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___15___conv_pwl_bn_running_var = self.getattr_L__mod___features___15___conv_pwl_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___15___conv_pwl_bn_weight = self.getattr_L__mod___features___15___conv_pwl_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___15___conv_pwl_bn_bias = self.getattr_L__mod___features___15___conv_pwl_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_322 = torch.nn.functional.batch_norm(x_321, getattr_l__mod___features___15___conv_pwl_bn_running_mean, getattr_l__mod___features___15___conv_pwl_bn_running_var, getattr_l__mod___features___15___conv_pwl_bn_weight, getattr_l__mod___features___15___conv_pwl_bn_bias, False, 0.1, 1e-05);  x_321 = getattr_l__mod___features___15___conv_pwl_bn_running_mean = getattr_l__mod___features___15___conv_pwl_bn_running_var = getattr_l__mod___features___15___conv_pwl_bn_weight = getattr_l__mod___features___15___conv_pwl_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_323 = self.getattr_L__mod___features___15___conv_pwl_bn_drop(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_326 = self.getattr_L__mod___features___15___conv_pwl_bn_act(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    getitem_20 = x_326[(slice(None, None, None), slice(0, 174, None))]
    add_10 = getitem_20 + shortcut_15;  getitem_20 = shortcut_15 = None
    getitem_21 = x_326[(slice(None, None, None), slice(174, None, None))];  x_326 = None
    x_327 = torch.cat([add_10, getitem_21], dim = 1);  add_10 = getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_328 = self.getattr_L__mod___features___16___conv(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___features___16___bn_running_mean = self.getattr_L__mod___features___16___bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___features___16___bn_running_var = self.getattr_L__mod___features___16___bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___features___16___bn_weight = self.getattr_L__mod___features___16___bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___features___16___bn_bias = self.getattr_L__mod___features___16___bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_329 = torch.nn.functional.batch_norm(x_328, getattr_l__mod___features___16___bn_running_mean, getattr_l__mod___features___16___bn_running_var, getattr_l__mod___features___16___bn_weight, getattr_l__mod___features___16___bn_bias, False, 0.1, 1e-05);  x_328 = getattr_l__mod___features___16___bn_running_mean = getattr_l__mod___features___16___bn_running_var = getattr_l__mod___features___16___bn_weight = getattr_l__mod___features___16___bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_330 = self.getattr_L__mod___features___16___bn_drop(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_334 = self.getattr_L__mod___features___16___bn_act(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_335 = self.L__mod___head_global_pool_pool(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_337 = self.L__mod___head_global_pool_flatten(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_338 = self.L__mod___head_drop(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_339 = self.L__mod___head_fc(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_340 = self.L__mod___head_flatten(x_339);  x_339 = None
    return (x_340,)
    