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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_5 = self.getattr_L__mod___stem___1___conv(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stem___1___bn_running_mean = self.getattr_L__mod___stem___1___bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stem___1___bn_running_var = self.getattr_L__mod___stem___1___bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stem___1___bn_weight = self.getattr_L__mod___stem___1___bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stem___1___bn_bias = self.getattr_L__mod___stem___1___bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_6 = torch.nn.functional.batch_norm(x_5, getattr_l__mod___stem___1___bn_running_mean, getattr_l__mod___stem___1___bn_running_var, getattr_l__mod___stem___1___bn_weight, getattr_l__mod___stem___1___bn_bias, False, 0.1, 1e-05);  x_5 = getattr_l__mod___stem___1___bn_running_mean = getattr_l__mod___stem___1___bn_running_var = getattr_l__mod___stem___1___bn_weight = getattr_l__mod___stem___1___bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_7 = self.getattr_L__mod___stem___1___bn_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_9 = self.getattr_L__mod___stem___1___bn_act(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_10 = self.getattr_L__mod___stem___2___conv(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stem___2___bn_running_mean = self.getattr_L__mod___stem___2___bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stem___2___bn_running_var = self.getattr_L__mod___stem___2___bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stem___2___bn_weight = self.getattr_L__mod___stem___2___bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stem___2___bn_bias = self.getattr_L__mod___stem___2___bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_11 = torch.nn.functional.batch_norm(x_10, getattr_l__mod___stem___2___bn_running_mean, getattr_l__mod___stem___2___bn_running_var, getattr_l__mod___stem___2___bn_weight, getattr_l__mod___stem___2___bn_bias, False, 0.1, 1e-05);  x_10 = getattr_l__mod___stem___2___bn_running_mean = getattr_l__mod___stem___2___bn_running_var = getattr_l__mod___stem___2___bn_weight = getattr_l__mod___stem___2___bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_12 = self.getattr_L__mod___stem___2___bn_drop(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_15 = self.getattr_L__mod___stem___2___bn_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_16 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_conv(x_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_17 = torch.nn.functional.batch_norm(x_16, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_16 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_18 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_20 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_0_bn_act(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_21 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_conv(x_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_22 = torch.nn.functional.batch_norm(x_21, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_21 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_23 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_drop(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_25 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_1_bn_act(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_26 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_conv(x_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_27 = torch.nn.functional.batch_norm(x_26, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_26 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_28 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_drop(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_30 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_2_bn_act(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_31 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_3_conv(x_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_32 = torch.nn.functional.batch_norm(x_31, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_bias, False, 0.1, 1e-05);  x_31 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_33 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_3_bn_drop(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_35 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_3_bn_act(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_36 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_4_conv(x_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_37 = torch.nn.functional.batch_norm(x_36, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_bias, False, 0.1, 1e-05);  x_36 = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_mid_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_38 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_4_bn_drop(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_40 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_mid_4_bn_act(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_42 = torch.cat([x_15, x_20, x_25, x_30, x_35, x_40], dim = 1);  x_15 = x_20 = x_25 = x_30 = x_35 = x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_43 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_conv(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_44 = torch.nn.functional.batch_norm(x_43, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_43 = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_45 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_drop(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_49 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv_concat_bn_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    x_50 = self.getattr_L__mod___stages___1___pool(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_51 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_conv(x_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_52 = torch.nn.functional.batch_norm(x_51, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_51 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_53 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_drop(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_55 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_0_bn_act(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_56 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_conv(x_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_57 = torch.nn.functional.batch_norm(x_56, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_56 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_58 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_drop(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_60 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_1_bn_act(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_61 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_conv(x_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_62 = torch.nn.functional.batch_norm(x_61, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_61 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_63 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_drop(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_65 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_2_bn_act(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_66 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_3_conv(x_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_67 = torch.nn.functional.batch_norm(x_66, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_bias, False, 0.1, 1e-05);  x_66 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_68 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_3_bn_drop(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_70 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_3_bn_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_71 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_4_conv(x_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_72 = torch.nn.functional.batch_norm(x_71, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_bias, False, 0.1, 1e-05);  x_71 = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_mid_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_73 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_4_bn_drop(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_75 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_mid_4_bn_act(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_77 = torch.cat([x_50, x_55, x_60, x_65, x_70, x_75], dim = 1);  x_50 = x_55 = x_60 = x_65 = x_70 = x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_78 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_conv(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_79 = torch.nn.functional.batch_norm(x_78, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_78 = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_80 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_drop(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_84 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv_concat_bn_act(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    x_85 = self.getattr_L__mod___stages___2___pool(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_86 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_conv(x_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_87 = torch.nn.functional.batch_norm(x_86, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_86 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_88 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_drop(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_90 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_0_bn_act(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_91 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_conv(x_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_92 = torch.nn.functional.batch_norm(x_91, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_91 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_93 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_drop(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_95 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_1_bn_act(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_96 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_conv(x_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_97 = torch.nn.functional.batch_norm(x_96, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_96 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_98 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_drop(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_100 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_2_bn_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_101 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_3_conv(x_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_102 = torch.nn.functional.batch_norm(x_101, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_bias, False, 0.1, 1e-05);  x_101 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_103 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_3_bn_drop(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_105 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_3_bn_act(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_106 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_4_conv(x_105)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_107 = torch.nn.functional.batch_norm(x_106, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_bias, False, 0.1, 1e-05);  x_106 = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_mid_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_108 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_4_bn_drop(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_110 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_mid_4_bn_act(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_112 = torch.cat([x_85, x_90, x_95, x_100, x_105, x_110], dim = 1);  x_85 = x_90 = x_95 = x_100 = x_105 = x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_113 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_conv(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_114 = torch.nn.functional.batch_norm(x_113, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_113 = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_115 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_drop(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_118 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv_concat_bn_act(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_119 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_0_conv(x_118)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_120 = torch.nn.functional.batch_norm(x_119, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_119 = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_121 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_0_bn_drop(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_123 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_0_bn_act(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_124 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_1_conv(x_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_125 = torch.nn.functional.batch_norm(x_124, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_124 = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_126 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_1_bn_drop(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_128 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_1_bn_act(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_129 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_2_conv(x_128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_130 = torch.nn.functional.batch_norm(x_129, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_129 = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_131 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_2_bn_drop(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_133 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_2_bn_act(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_134 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_3_conv(x_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_135 = torch.nn.functional.batch_norm(x_134, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_bias, False, 0.1, 1e-05);  x_134 = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_136 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_3_bn_drop(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_138 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_3_bn_act(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_139 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_4_conv(x_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_140 = torch.nn.functional.batch_norm(x_139, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_bias, False, 0.1, 1e-05);  x_139 = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv_mid_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_141 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_4_bn_drop(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_143 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_mid_4_bn_act(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_145 = torch.cat([x_118, x_123, x_128, x_133, x_138, x_143], dim = 1);  x_118 = x_123 = x_128 = x_133 = x_138 = x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_146 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_concat_conv(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_147 = torch.nn.functional.batch_norm(x_146, getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_bias, False, 0.1, 1e-05);  x_146 = getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_148 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_concat_bn_drop(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_152 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv_concat_bn_act(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    x_153 = self.getattr_L__mod___stages___3___pool(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_154 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_conv(x_153)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_155 = torch.nn.functional.batch_norm(x_154, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_154 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_156 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_drop(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_158 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_0_bn_act(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_159 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_conv(x_158)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_160 = torch.nn.functional.batch_norm(x_159, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_159 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_161 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_drop(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_163 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_1_bn_act(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_164 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_conv(x_163)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_165 = torch.nn.functional.batch_norm(x_164, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_164 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_166 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_drop(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_168 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_2_bn_act(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_169 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_3_conv(x_168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_170 = torch.nn.functional.batch_norm(x_169, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_bias, False, 0.1, 1e-05);  x_169 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_171 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_3_bn_drop(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_173 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_3_bn_act(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_174 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_4_conv(x_173)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_175 = torch.nn.functional.batch_norm(x_174, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_bias, False, 0.1, 1e-05);  x_174 = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_mid_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_176 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_4_bn_drop(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_178 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_mid_4_bn_act(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_180 = torch.cat([x_153, x_158, x_163, x_168, x_173, x_178], dim = 1);  x_153 = x_158 = x_163 = x_168 = x_173 = x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_181 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_conv(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_182 = torch.nn.functional.batch_norm(x_181, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_bias, False, 0.1, 1e-05);  x_181 = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_183 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_drop(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_186 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv_concat_bn_act(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_187 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_0_conv(x_186)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_0_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_0_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_0_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_0_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_188 = torch.nn.functional.batch_norm(x_187, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_bias, False, 0.1, 1e-05);  x_187 = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_0_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_189 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_0_bn_drop(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_191 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_0_bn_act(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_192 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_1_conv(x_191)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_193 = torch.nn.functional.batch_norm(x_192, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_bias, False, 0.1, 1e-05);  x_192 = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_194 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_1_bn_drop(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_196 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_1_bn_act(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_197 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_2_conv(x_196)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_198 = torch.nn.functional.batch_norm(x_197, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_bias, False, 0.1, 1e-05);  x_197 = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_199 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_2_bn_drop(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_201 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_2_bn_act(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_202 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_3_conv(x_201)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_203 = torch.nn.functional.batch_norm(x_202, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_bias, False, 0.1, 1e-05);  x_202 = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_204 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_3_bn_drop(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_206 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_3_bn_act(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_207 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_4_conv(x_206)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_208 = torch.nn.functional.batch_norm(x_207, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_bias, False, 0.1, 1e-05);  x_207 = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv_mid_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_209 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_4_bn_drop(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_211 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_mid_4_bn_act(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    x_213 = torch.cat([x_186, x_191, x_196, x_201, x_206, x_211], dim = 1);  x_186 = x_191 = x_196 = x_201 = x_206 = x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_214 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_concat_conv(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_concat_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_concat_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_concat_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_concat_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_215 = torch.nn.functional.batch_norm(x_214, getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_bias, False, 0.1, 1e-05);  x_214 = getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv_concat_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_216 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_concat_bn_drop(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_221 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv_concat_bn_act(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_222 = self.L__mod___head_global_pool_pool(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_224 = self.L__mod___head_global_pool_flatten(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_225 = self.L__mod___head_drop(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_226 = self.L__mod___head_fc(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_227 = self.L__mod___head_flatten(x_226);  x_226 = None
    return (x_227,)
    