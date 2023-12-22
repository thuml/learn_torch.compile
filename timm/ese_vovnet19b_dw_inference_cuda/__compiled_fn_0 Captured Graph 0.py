from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.getattr_L__mod___stem___0___conv(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stem___0___bn_running_mean = self.getattr_L__mod___stem___0___bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stem___0___bn_running_var = self.getattr_L__mod___stem___0___bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stem___0___bn_weight = self.getattr_L__mod___stem___0___bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stem___0___bn_bias = self.getattr_L__mod___stem___0___bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, getattr_l__mod___stem___0___bn_running_mean, getattr_l__mod___stem___0___bn_running_var, getattr_l__mod___stem___0___bn_weight, getattr_l__mod___stem___0___bn_bias, False, 0.1, 1e-05);  x = getattr_l__mod___stem___0___bn_running_mean = getattr_l__mod___stem___0___bn_running_var = getattr_l__mod___stem___0___bn_weight = getattr_l__mod___stem___0___bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.getattr_L__mod___stem___0___bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_4 = self.getattr_L__mod___stem___0___bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_5 = self.getattr_L__mod___stem___1___conv_dw(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_6 = self.getattr_L__mod___stem___1___conv_pw(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stem___1___bn_running_mean = self.getattr_L__mod___stem___1___bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stem___1___bn_running_var = self.getattr_L__mod___stem___1___bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stem___1___bn_weight = self.getattr_L__mod___stem___1___bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stem___1___bn_bias = self.getattr_L__mod___stem___1___bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, getattr_l__mod___stem___1___bn_running_mean, getattr_l__mod___stem___1___bn_running_var, getattr_l__mod___stem___1___bn_weight, getattr_l__mod___stem___1___bn_bias, False, 0.1, 1e-05);  x_6 = getattr_l__mod___stem___1___bn_running_mean = getattr_l__mod___stem___1___bn_running_var = getattr_l__mod___stem___1___bn_weight = getattr_l__mod___stem___1___bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.getattr_L__mod___stem___1___bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_10 = self.getattr_L__mod___stem___1___bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_11 = self.getattr_L__mod___stem___2___conv_dw(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_12 = self.getattr_L__mod___stem___2___conv_pw(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stem___2___bn_running_mean = self.getattr_L__mod___stem___2___bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stem___2___bn_running_var = self.getattr_L__mod___stem___2___bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stem___2___bn_weight = self.getattr_L__mod___stem___2___bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stem___2___bn_bias = self.getattr_L__mod___stem___2___bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_13 = torch.nn.functional.batch_norm(x_12, getattr_l__mod___stem___2___bn_running_mean, getattr_l__mod___stem___2___bn_running_var, getattr_l__mod___stem___2___bn_weight, getattr_l__mod___stem___2___bn_bias, False, 0.1, 1e-05);  x_12 = getattr_l__mod___stem___2___bn_running_mean = getattr_l__mod___stem___2___bn_running_var = getattr_l__mod___stem___2___bn_weight = getattr_l__mod___stem___2___bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_14 = self.getattr_L__mod___stem___2___bn_drop(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_17 = self.getattr_L__mod___stem___2___bn_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_18 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_reduction_conv(x_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_reduction_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_reduction_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_reduction_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_reduction_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_19 = torch.nn.functional.batch_norm(x_18, getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_bias, False, 0.1, 1e-05);  x_18 = getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_reduction_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_20 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_reduction_bn_drop(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_23 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_reduction_bn_act(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_24 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_conv_dw(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_25 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_conv_pw(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_26 = torch.nn.functional.batch_norm(x_25, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_25 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_27 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_drop(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_29 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_act(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_30 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_conv_dw(x_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_31 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_conv_pw(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_32 = torch.nn.functional.batch_norm(x_31, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_31 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_33 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_drop(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_35 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_act(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_36 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_conv_dw(x_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_37 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_conv_pw(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_38 = torch.nn.functional.batch_norm(x_37, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_37 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_39 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_drop(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_41 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_act(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_43 = torch.cat([x_17, x_29, x_35, x_41], dim = 1);  x_17 = x_29 = x_35 = x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_44 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_conv(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_45 = torch.nn.functional.batch_norm(x_44, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_44 = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_46 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_drop(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_49 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_act(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_49.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    x_se_1 = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_fc(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___stages___0___blocks___0___attn_gate = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_gate(x_se_1);  x_se_1 = None
    x_51 = x_49 * getattr_getattr_l__mod___stages___0___blocks___0___attn_gate;  x_49 = getattr_getattr_l__mod___stages___0___blocks___0___attn_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    x_52 = self.getattr_L__mod___stages___1___pool(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_53 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_reduction_conv(x_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_reduction_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_reduction_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_reduction_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_reduction_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_54 = torch.nn.functional.batch_norm(x_53, getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_bias, False, 0.1, 1e-05);  x_53 = getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_reduction_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_55 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_reduction_bn_drop(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_58 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_reduction_bn_act(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_59 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_conv_dw(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_60 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_conv_pw(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_61 = torch.nn.functional.batch_norm(x_60, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_60 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_62 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_drop(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_64 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_act(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_65 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_conv_dw(x_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_66 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_conv_pw(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_67 = torch.nn.functional.batch_norm(x_66, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_66 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_68 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_drop(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_70 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_71 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_conv_dw(x_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_72 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_conv_pw(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_73 = torch.nn.functional.batch_norm(x_72, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_72 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_74 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_drop(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_76 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_act(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_78 = torch.cat([x_52, x_64, x_70, x_76], dim = 1);  x_52 = x_64 = x_70 = x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_79 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_conv(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_80 = torch.nn.functional.batch_norm(x_79, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_79 = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_81 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_drop(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_84 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_2 = x_84.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    x_se_3 = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_fc(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___stages___1___blocks___0___attn_gate = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_gate(x_se_3);  x_se_3 = None
    x_86 = x_84 * getattr_getattr_l__mod___stages___1___blocks___0___attn_gate;  x_84 = getattr_getattr_l__mod___stages___1___blocks___0___attn_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    x_87 = self.getattr_L__mod___stages___2___pool(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_88 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_reduction_conv(x_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_reduction_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_reduction_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_reduction_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_reduction_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_89 = torch.nn.functional.batch_norm(x_88, getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_bias, False, 0.1, 1e-05);  x_88 = getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_reduction_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_90 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_reduction_bn_drop(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_93 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_reduction_bn_act(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_94 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_conv_dw(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_95 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_conv_pw(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_96 = torch.nn.functional.batch_norm(x_95, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_95 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_97 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_drop(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_99 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_act(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_100 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_conv_dw(x_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_101 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_conv_pw(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_102 = torch.nn.functional.batch_norm(x_101, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_101 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_103 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_drop(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_105 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_act(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_106 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_conv_dw(x_105)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_107 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_conv_pw(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_108 = torch.nn.functional.batch_norm(x_107, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_107 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_109 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_drop(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_111 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_act(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_113 = torch.cat([x_87, x_99, x_105, x_111], dim = 1);  x_87 = x_99 = x_105 = x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_114 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_conv(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_115 = torch.nn.functional.batch_norm(x_114, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_114 = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_116 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_drop(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_119 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_119.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    x_se_5 = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_fc(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___stages___2___blocks___0___attn_gate = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_gate(x_se_5);  x_se_5 = None
    x_121 = x_119 * getattr_getattr_l__mod___stages___2___blocks___0___attn_gate;  x_119 = getattr_getattr_l__mod___stages___2___blocks___0___attn_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    x_122 = self.getattr_L__mod___stages___3___pool(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_123 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_reduction_conv(x_122)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_reduction_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_reduction_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_reduction_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_reduction_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_124 = torch.nn.functional.batch_norm(x_123, getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_bias, False, 0.1, 1e-05);  x_123 = getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_reduction_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_125 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_reduction_bn_drop(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_128 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_reduction_bn_act(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_129 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_conv_dw(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_130 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_conv_pw(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_131 = torch.nn.functional.batch_norm(x_130, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_130 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_132 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_drop(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_134 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_act(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_135 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_conv_dw(x_134)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_136 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_conv_pw(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_137 = torch.nn.functional.batch_norm(x_136, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_136 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_138 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_drop(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_140 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_act(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:42, code: x = self.conv_dw(x)
    x_141 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_conv_dw(x_140)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/separable_conv.py:43, code: x = self.conv_pw(x)
    x_142 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_conv_pw(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_143 = torch.nn.functional.batch_norm(x_142, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_142 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_144 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_drop(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_146 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_act(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_148 = torch.cat([x_122, x_134, x_140, x_146], dim = 1);  x_122 = x_134 = x_140 = x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_149 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_conv(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_150 = torch.nn.functional.batch_norm(x_149, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_149 = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_151 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_drop(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_154 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:66, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_6 = x_154.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:70, code: x_se = self.fc(x_se)
    x_se_7 = self.getattr_getattr_L__mod___stages___3___blocks___0___attn_fc(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:71, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___stages___3___blocks___0___attn_gate = self.getattr_getattr_L__mod___stages___3___blocks___0___attn_gate(x_se_7);  x_se_7 = None
    x_157 = x_154 * getattr_getattr_l__mod___stages___3___blocks___0___attn_gate;  x_154 = getattr_getattr_l__mod___stages___3___blocks___0___attn_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_158 = self.L__mod___head_global_pool_pool(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_160 = self.L__mod___head_global_pool_flatten(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_161 = self.L__mod___head_drop(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_162 = self.L__mod___head_fc(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_163 = self.L__mod___head_flatten(x_162);  x_162 = None
    return (x_163,)
    