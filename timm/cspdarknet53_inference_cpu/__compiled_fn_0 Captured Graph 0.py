from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___stem_conv1_conv(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___stem_conv1_bn_running_mean = self.L__mod___stem_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv1_bn_running_var = self.L__mod___stem_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv1_bn_weight = self.L__mod___stem_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv1_bn_bias = self.L__mod___stem_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___stem_conv1_bn_running_mean, l__mod___stem_conv1_bn_running_var, l__mod___stem_conv1_bn_weight, l__mod___stem_conv1_bn_bias, False, 0.1, 1e-05);  x = l__mod___stem_conv1_bn_running_mean = l__mod___stem_conv1_bn_running_var = l__mod___stem_conv1_bn_weight = l__mod___stem_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___stem_conv1_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_5 = self.L__mod___stem_conv1_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    x_6 = self.getattr_L__mod___stages___0___conv_down_conv(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___0___conv_down_bn_running_mean = self.getattr_L__mod___stages___0___conv_down_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___0___conv_down_bn_running_var = self.getattr_L__mod___stages___0___conv_down_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___0___conv_down_bn_weight = self.getattr_L__mod___stages___0___conv_down_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___0___conv_down_bn_bias = self.getattr_L__mod___stages___0___conv_down_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, getattr_l__mod___stages___0___conv_down_bn_running_mean, getattr_l__mod___stages___0___conv_down_bn_running_var, getattr_l__mod___stages___0___conv_down_bn_weight, getattr_l__mod___stages___0___conv_down_bn_bias, False, 0.1, 1e-05);  x_6 = getattr_l__mod___stages___0___conv_down_bn_running_mean = getattr_l__mod___stages___0___conv_down_bn_running_var = getattr_l__mod___stages___0___conv_down_bn_weight = getattr_l__mod___stages___0___conv_down_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.getattr_L__mod___stages___0___conv_down_bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_10 = self.getattr_L__mod___stages___0___conv_down_bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:128, code: x = self.aa(x)
    x_12 = self.getattr_L__mod___stages___0___conv_down_aa(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_13 = self.getattr_L__mod___stages___0___conv_exp_conv(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___0___conv_exp_bn_running_mean = self.getattr_L__mod___stages___0___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___0___conv_exp_bn_running_var = self.getattr_L__mod___stages___0___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___0___conv_exp_bn_weight = self.getattr_L__mod___stages___0___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___0___conv_exp_bn_bias = self.getattr_L__mod___stages___0___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_14 = torch.nn.functional.batch_norm(x_13, getattr_l__mod___stages___0___conv_exp_bn_running_mean, getattr_l__mod___stages___0___conv_exp_bn_running_var, getattr_l__mod___stages___0___conv_exp_bn_weight, getattr_l__mod___stages___0___conv_exp_bn_bias, False, 0.1, 1e-05);  x_13 = getattr_l__mod___stages___0___conv_exp_bn_running_mean = getattr_l__mod___stages___0___conv_exp_bn_running_var = getattr_l__mod___stages___0___conv_exp_bn_weight = getattr_l__mod___stages___0___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_15 = self.getattr_L__mod___stages___0___conv_exp_bn_drop(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_18 = self.getattr_L__mod___stages___0___conv_exp_bn_act(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    split = x_18.split(64, dim = 1);  x_18 = None
    xs = split[0]
    shortcut = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_19 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv1_conv(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_20 = torch.nn.functional.batch_norm(x_19, getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_bias, False, 0.1, 1e-05);  x_19 = getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_21 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv1_bn_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_24 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv1_bn_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_25 = self.getattr_getattr_L__mod___stages___0___blocks___0___attn(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_26 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv2_conv(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___0___blocks___0___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___0___blocks___0___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_27 = torch.nn.functional.batch_norm(x_26, getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_running_mean, getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_running_var, getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_weight, getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_bias, False, 0.1, 1e-05);  x_26 = getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_running_mean = getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_running_var = getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_weight = getattr_getattr_l__mod___stages___0___blocks___0___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_28 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv2_bn_drop(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_31 = self.getattr_getattr_L__mod___stages___0___blocks___0___conv2_bn_act(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___0___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___0___blocks___0___drop_path(x_31);  x_31 = None
    xb_1 = getattr_getattr_l__mod___stages___0___blocks___0___drop_path + shortcut;  getattr_getattr_l__mod___stages___0___blocks___0___drop_path = shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_33 = self.getattr_L__mod___stages___0___conv_transition_b_conv(xb_1);  xb_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___0___conv_transition_b_bn_running_mean = self.getattr_L__mod___stages___0___conv_transition_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___0___conv_transition_b_bn_running_var = self.getattr_L__mod___stages___0___conv_transition_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___0___conv_transition_b_bn_weight = self.getattr_L__mod___stages___0___conv_transition_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___0___conv_transition_b_bn_bias = self.getattr_L__mod___stages___0___conv_transition_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_34 = torch.nn.functional.batch_norm(x_33, getattr_l__mod___stages___0___conv_transition_b_bn_running_mean, getattr_l__mod___stages___0___conv_transition_b_bn_running_var, getattr_l__mod___stages___0___conv_transition_b_bn_weight, getattr_l__mod___stages___0___conv_transition_b_bn_bias, False, 0.1, 1e-05);  x_33 = getattr_l__mod___stages___0___conv_transition_b_bn_running_mean = getattr_l__mod___stages___0___conv_transition_b_bn_running_var = getattr_l__mod___stages___0___conv_transition_b_bn_weight = getattr_l__mod___stages___0___conv_transition_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_35 = self.getattr_L__mod___stages___0___conv_transition_b_bn_drop(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_37 = self.getattr_L__mod___stages___0___conv_transition_b_bn_act(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:338, code: xb = self.conv_transition_b(xb).contiguous()
    xb_2 = x_37.contiguous();  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    cat = torch.cat([xs, xb_2], dim = 1);  xs = xb_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_38 = self.getattr_L__mod___stages___0___conv_transition_conv(cat);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___0___conv_transition_bn_running_mean = self.getattr_L__mod___stages___0___conv_transition_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___0___conv_transition_bn_running_var = self.getattr_L__mod___stages___0___conv_transition_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___0___conv_transition_bn_weight = self.getattr_L__mod___stages___0___conv_transition_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___0___conv_transition_bn_bias = self.getattr_L__mod___stages___0___conv_transition_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_39 = torch.nn.functional.batch_norm(x_38, getattr_l__mod___stages___0___conv_transition_bn_running_mean, getattr_l__mod___stages___0___conv_transition_bn_running_var, getattr_l__mod___stages___0___conv_transition_bn_weight, getattr_l__mod___stages___0___conv_transition_bn_bias, False, 0.1, 1e-05);  x_38 = getattr_l__mod___stages___0___conv_transition_bn_running_mean = getattr_l__mod___stages___0___conv_transition_bn_running_var = getattr_l__mod___stages___0___conv_transition_bn_weight = getattr_l__mod___stages___0___conv_transition_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_40 = self.getattr_L__mod___stages___0___conv_transition_bn_drop(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    out = self.getattr_L__mod___stages___0___conv_transition_bn_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    x_43 = self.getattr_L__mod___stages___1___conv_down_conv(out);  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___1___conv_down_bn_running_mean = self.getattr_L__mod___stages___1___conv_down_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___1___conv_down_bn_running_var = self.getattr_L__mod___stages___1___conv_down_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___1___conv_down_bn_weight = self.getattr_L__mod___stages___1___conv_down_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___1___conv_down_bn_bias = self.getattr_L__mod___stages___1___conv_down_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_44 = torch.nn.functional.batch_norm(x_43, getattr_l__mod___stages___1___conv_down_bn_running_mean, getattr_l__mod___stages___1___conv_down_bn_running_var, getattr_l__mod___stages___1___conv_down_bn_weight, getattr_l__mod___stages___1___conv_down_bn_bias, False, 0.1, 1e-05);  x_43 = getattr_l__mod___stages___1___conv_down_bn_running_mean = getattr_l__mod___stages___1___conv_down_bn_running_var = getattr_l__mod___stages___1___conv_down_bn_weight = getattr_l__mod___stages___1___conv_down_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_45 = self.getattr_L__mod___stages___1___conv_down_bn_drop(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_47 = self.getattr_L__mod___stages___1___conv_down_bn_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:128, code: x = self.aa(x)
    x_49 = self.getattr_L__mod___stages___1___conv_down_aa(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_50 = self.getattr_L__mod___stages___1___conv_exp_conv(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___1___conv_exp_bn_running_mean = self.getattr_L__mod___stages___1___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___1___conv_exp_bn_running_var = self.getattr_L__mod___stages___1___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___1___conv_exp_bn_weight = self.getattr_L__mod___stages___1___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___1___conv_exp_bn_bias = self.getattr_L__mod___stages___1___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_51 = torch.nn.functional.batch_norm(x_50, getattr_l__mod___stages___1___conv_exp_bn_running_mean, getattr_l__mod___stages___1___conv_exp_bn_running_var, getattr_l__mod___stages___1___conv_exp_bn_weight, getattr_l__mod___stages___1___conv_exp_bn_bias, False, 0.1, 1e-05);  x_50 = getattr_l__mod___stages___1___conv_exp_bn_running_mean = getattr_l__mod___stages___1___conv_exp_bn_running_var = getattr_l__mod___stages___1___conv_exp_bn_weight = getattr_l__mod___stages___1___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_52 = self.getattr_L__mod___stages___1___conv_exp_bn_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_55 = self.getattr_L__mod___stages___1___conv_exp_bn_act(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    split_1 = x_55.split(64, dim = 1);  x_55 = None
    xs_1 = split_1[0]
    shortcut_1 = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_56 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv1_conv(shortcut_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_57 = torch.nn.functional.batch_norm(x_56, getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_bias, False, 0.1, 1e-05);  x_56 = getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_58 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv1_bn_drop(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_61 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv1_bn_act(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_62 = self.getattr_getattr_L__mod___stages___1___blocks___0___attn(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_63 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv2_conv(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___0___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___0___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_64 = torch.nn.functional.batch_norm(x_63, getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_weight, getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_bias, False, 0.1, 1e-05);  x_63 = getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_weight = getattr_getattr_l__mod___stages___1___blocks___0___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_65 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv2_bn_drop(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_68 = self.getattr_getattr_L__mod___stages___1___blocks___0___conv2_bn_act(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___1___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___1___blocks___0___drop_path(x_68);  x_68 = None
    shortcut_2 = getattr_getattr_l__mod___stages___1___blocks___0___drop_path + shortcut_1;  getattr_getattr_l__mod___stages___1___blocks___0___drop_path = shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_70 = self.getattr_getattr_L__mod___stages___1___blocks___1___conv1_conv(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___1___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___1___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___1___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___1___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_71 = torch.nn.functional.batch_norm(x_70, getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_weight, getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_bias, False, 0.1, 1e-05);  x_70 = getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_weight = getattr_getattr_l__mod___stages___1___blocks___1___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_72 = self.getattr_getattr_L__mod___stages___1___blocks___1___conv1_bn_drop(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_75 = self.getattr_getattr_L__mod___stages___1___blocks___1___conv1_bn_act(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_76 = self.getattr_getattr_L__mod___stages___1___blocks___1___attn(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_77 = self.getattr_getattr_L__mod___stages___1___blocks___1___conv2_conv(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___1___blocks___1___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___1___blocks___1___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_weight = self.getattr_getattr_L__mod___stages___1___blocks___1___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_bias = self.getattr_getattr_L__mod___stages___1___blocks___1___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_78 = torch.nn.functional.batch_norm(x_77, getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_running_mean, getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_running_var, getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_weight, getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_bias, False, 0.1, 1e-05);  x_77 = getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_running_mean = getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_running_var = getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_weight = getattr_getattr_l__mod___stages___1___blocks___1___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_79 = self.getattr_getattr_L__mod___stages___1___blocks___1___conv2_bn_drop(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_82 = self.getattr_getattr_L__mod___stages___1___blocks___1___conv2_bn_act(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___1___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___1___blocks___1___drop_path(x_82);  x_82 = None
    xb_4 = getattr_getattr_l__mod___stages___1___blocks___1___drop_path + shortcut_2;  getattr_getattr_l__mod___stages___1___blocks___1___drop_path = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_84 = self.getattr_L__mod___stages___1___conv_transition_b_conv(xb_4);  xb_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___1___conv_transition_b_bn_running_mean = self.getattr_L__mod___stages___1___conv_transition_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___1___conv_transition_b_bn_running_var = self.getattr_L__mod___stages___1___conv_transition_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___1___conv_transition_b_bn_weight = self.getattr_L__mod___stages___1___conv_transition_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___1___conv_transition_b_bn_bias = self.getattr_L__mod___stages___1___conv_transition_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_85 = torch.nn.functional.batch_norm(x_84, getattr_l__mod___stages___1___conv_transition_b_bn_running_mean, getattr_l__mod___stages___1___conv_transition_b_bn_running_var, getattr_l__mod___stages___1___conv_transition_b_bn_weight, getattr_l__mod___stages___1___conv_transition_b_bn_bias, False, 0.1, 1e-05);  x_84 = getattr_l__mod___stages___1___conv_transition_b_bn_running_mean = getattr_l__mod___stages___1___conv_transition_b_bn_running_var = getattr_l__mod___stages___1___conv_transition_b_bn_weight = getattr_l__mod___stages___1___conv_transition_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_86 = self.getattr_L__mod___stages___1___conv_transition_b_bn_drop(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_88 = self.getattr_L__mod___stages___1___conv_transition_b_bn_act(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:338, code: xb = self.conv_transition_b(xb).contiguous()
    xb_5 = x_88.contiguous();  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    cat_1 = torch.cat([xs_1, xb_5], dim = 1);  xs_1 = xb_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_89 = self.getattr_L__mod___stages___1___conv_transition_conv(cat_1);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___1___conv_transition_bn_running_mean = self.getattr_L__mod___stages___1___conv_transition_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___1___conv_transition_bn_running_var = self.getattr_L__mod___stages___1___conv_transition_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___1___conv_transition_bn_weight = self.getattr_L__mod___stages___1___conv_transition_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___1___conv_transition_bn_bias = self.getattr_L__mod___stages___1___conv_transition_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_90 = torch.nn.functional.batch_norm(x_89, getattr_l__mod___stages___1___conv_transition_bn_running_mean, getattr_l__mod___stages___1___conv_transition_bn_running_var, getattr_l__mod___stages___1___conv_transition_bn_weight, getattr_l__mod___stages___1___conv_transition_bn_bias, False, 0.1, 1e-05);  x_89 = getattr_l__mod___stages___1___conv_transition_bn_running_mean = getattr_l__mod___stages___1___conv_transition_bn_running_var = getattr_l__mod___stages___1___conv_transition_bn_weight = getattr_l__mod___stages___1___conv_transition_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_91 = self.getattr_L__mod___stages___1___conv_transition_bn_drop(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    out_1 = self.getattr_L__mod___stages___1___conv_transition_bn_act(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    x_94 = self.getattr_L__mod___stages___2___conv_down_conv(out_1);  out_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___2___conv_down_bn_running_mean = self.getattr_L__mod___stages___2___conv_down_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___2___conv_down_bn_running_var = self.getattr_L__mod___stages___2___conv_down_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___2___conv_down_bn_weight = self.getattr_L__mod___stages___2___conv_down_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___2___conv_down_bn_bias = self.getattr_L__mod___stages___2___conv_down_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_95 = torch.nn.functional.batch_norm(x_94, getattr_l__mod___stages___2___conv_down_bn_running_mean, getattr_l__mod___stages___2___conv_down_bn_running_var, getattr_l__mod___stages___2___conv_down_bn_weight, getattr_l__mod___stages___2___conv_down_bn_bias, False, 0.1, 1e-05);  x_94 = getattr_l__mod___stages___2___conv_down_bn_running_mean = getattr_l__mod___stages___2___conv_down_bn_running_var = getattr_l__mod___stages___2___conv_down_bn_weight = getattr_l__mod___stages___2___conv_down_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_96 = self.getattr_L__mod___stages___2___conv_down_bn_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_98 = self.getattr_L__mod___stages___2___conv_down_bn_act(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:128, code: x = self.aa(x)
    x_100 = self.getattr_L__mod___stages___2___conv_down_aa(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_101 = self.getattr_L__mod___stages___2___conv_exp_conv(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___2___conv_exp_bn_running_mean = self.getattr_L__mod___stages___2___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___2___conv_exp_bn_running_var = self.getattr_L__mod___stages___2___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___2___conv_exp_bn_weight = self.getattr_L__mod___stages___2___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___2___conv_exp_bn_bias = self.getattr_L__mod___stages___2___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_102 = torch.nn.functional.batch_norm(x_101, getattr_l__mod___stages___2___conv_exp_bn_running_mean, getattr_l__mod___stages___2___conv_exp_bn_running_var, getattr_l__mod___stages___2___conv_exp_bn_weight, getattr_l__mod___stages___2___conv_exp_bn_bias, False, 0.1, 1e-05);  x_101 = getattr_l__mod___stages___2___conv_exp_bn_running_mean = getattr_l__mod___stages___2___conv_exp_bn_running_var = getattr_l__mod___stages___2___conv_exp_bn_weight = getattr_l__mod___stages___2___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_103 = self.getattr_L__mod___stages___2___conv_exp_bn_drop(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_106 = self.getattr_L__mod___stages___2___conv_exp_bn_act(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    split_2 = x_106.split(128, dim = 1);  x_106 = None
    xs_2 = split_2[0]
    shortcut_3 = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_107 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv1_conv(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_108 = torch.nn.functional.batch_norm(x_107, getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_bias, False, 0.1, 1e-05);  x_107 = getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_109 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv1_bn_drop(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_112 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv1_bn_act(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_113 = self.getattr_getattr_L__mod___stages___2___blocks___0___attn(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_114 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv2_conv(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___0___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___0___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_115 = torch.nn.functional.batch_norm(x_114, getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_bias, False, 0.1, 1e-05);  x_114 = getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___0___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_116 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv2_bn_drop(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_119 = self.getattr_getattr_L__mod___stages___2___blocks___0___conv2_bn_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___0___drop_path(x_119);  x_119 = None
    shortcut_4 = getattr_getattr_l__mod___stages___2___blocks___0___drop_path + shortcut_3;  getattr_getattr_l__mod___stages___2___blocks___0___drop_path = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_121 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv1_conv(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_122 = torch.nn.functional.batch_norm(x_121, getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_bias, False, 0.1, 1e-05);  x_121 = getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_123 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv1_bn_drop(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_126 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv1_bn_act(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_127 = self.getattr_getattr_L__mod___stages___2___blocks___1___attn(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_128 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv2_conv(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___1___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___1___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_129 = torch.nn.functional.batch_norm(x_128, getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_bias, False, 0.1, 1e-05);  x_128 = getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___1___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_130 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv2_bn_drop(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_133 = self.getattr_getattr_L__mod___stages___2___blocks___1___conv2_bn_act(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___1___drop_path(x_133);  x_133 = None
    shortcut_5 = getattr_getattr_l__mod___stages___2___blocks___1___drop_path + shortcut_4;  getattr_getattr_l__mod___stages___2___blocks___1___drop_path = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_135 = self.getattr_getattr_L__mod___stages___2___blocks___2___conv1_conv(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___2___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___2___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___2___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___2___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_136 = torch.nn.functional.batch_norm(x_135, getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_bias, False, 0.1, 1e-05);  x_135 = getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___2___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_137 = self.getattr_getattr_L__mod___stages___2___blocks___2___conv1_bn_drop(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_140 = self.getattr_getattr_L__mod___stages___2___blocks___2___conv1_bn_act(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_141 = self.getattr_getattr_L__mod___stages___2___blocks___2___attn(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_142 = self.getattr_getattr_L__mod___stages___2___blocks___2___conv2_conv(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___2___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___2___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___2___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___2___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_143 = torch.nn.functional.batch_norm(x_142, getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_bias, False, 0.1, 1e-05);  x_142 = getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___2___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_144 = self.getattr_getattr_L__mod___stages___2___blocks___2___conv2_bn_drop(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_147 = self.getattr_getattr_L__mod___stages___2___blocks___2___conv2_bn_act(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___2___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___2___drop_path(x_147);  x_147 = None
    shortcut_6 = getattr_getattr_l__mod___stages___2___blocks___2___drop_path + shortcut_5;  getattr_getattr_l__mod___stages___2___blocks___2___drop_path = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_149 = self.getattr_getattr_L__mod___stages___2___blocks___3___conv1_conv(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___3___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___3___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___3___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___3___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_150 = torch.nn.functional.batch_norm(x_149, getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_bias, False, 0.1, 1e-05);  x_149 = getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___3___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_151 = self.getattr_getattr_L__mod___stages___2___blocks___3___conv1_bn_drop(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_154 = self.getattr_getattr_L__mod___stages___2___blocks___3___conv1_bn_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_155 = self.getattr_getattr_L__mod___stages___2___blocks___3___attn(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_156 = self.getattr_getattr_L__mod___stages___2___blocks___3___conv2_conv(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___3___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___3___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___3___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___3___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_157 = torch.nn.functional.batch_norm(x_156, getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_bias, False, 0.1, 1e-05);  x_156 = getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___3___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_158 = self.getattr_getattr_L__mod___stages___2___blocks___3___conv2_bn_drop(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_161 = self.getattr_getattr_L__mod___stages___2___blocks___3___conv2_bn_act(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___3___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___3___drop_path(x_161);  x_161 = None
    shortcut_7 = getattr_getattr_l__mod___stages___2___blocks___3___drop_path + shortcut_6;  getattr_getattr_l__mod___stages___2___blocks___3___drop_path = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_163 = self.getattr_getattr_L__mod___stages___2___blocks___4___conv1_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___4___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___4___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___4___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___4___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_164 = torch.nn.functional.batch_norm(x_163, getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_bias, False, 0.1, 1e-05);  x_163 = getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___4___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_165 = self.getattr_getattr_L__mod___stages___2___blocks___4___conv1_bn_drop(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_168 = self.getattr_getattr_L__mod___stages___2___blocks___4___conv1_bn_act(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_169 = self.getattr_getattr_L__mod___stages___2___blocks___4___attn(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_170 = self.getattr_getattr_L__mod___stages___2___blocks___4___conv2_conv(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___4___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___4___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___4___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___4___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_171 = torch.nn.functional.batch_norm(x_170, getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_bias, False, 0.1, 1e-05);  x_170 = getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___4___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_172 = self.getattr_getattr_L__mod___stages___2___blocks___4___conv2_bn_drop(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_175 = self.getattr_getattr_L__mod___stages___2___blocks___4___conv2_bn_act(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___4___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___4___drop_path(x_175);  x_175 = None
    shortcut_8 = getattr_getattr_l__mod___stages___2___blocks___4___drop_path + shortcut_7;  getattr_getattr_l__mod___stages___2___blocks___4___drop_path = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_177 = self.getattr_getattr_L__mod___stages___2___blocks___5___conv1_conv(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___5___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___5___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___5___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___5___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_178 = torch.nn.functional.batch_norm(x_177, getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_bias, False, 0.1, 1e-05);  x_177 = getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___5___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_179 = self.getattr_getattr_L__mod___stages___2___blocks___5___conv1_bn_drop(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_182 = self.getattr_getattr_L__mod___stages___2___blocks___5___conv1_bn_act(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_183 = self.getattr_getattr_L__mod___stages___2___blocks___5___attn(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_184 = self.getattr_getattr_L__mod___stages___2___blocks___5___conv2_conv(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___5___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___5___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___5___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___5___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_185 = torch.nn.functional.batch_norm(x_184, getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_bias, False, 0.1, 1e-05);  x_184 = getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___5___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_186 = self.getattr_getattr_L__mod___stages___2___blocks___5___conv2_bn_drop(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_189 = self.getattr_getattr_L__mod___stages___2___blocks___5___conv2_bn_act(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___5___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___5___drop_path(x_189);  x_189 = None
    shortcut_9 = getattr_getattr_l__mod___stages___2___blocks___5___drop_path + shortcut_8;  getattr_getattr_l__mod___stages___2___blocks___5___drop_path = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_191 = self.getattr_getattr_L__mod___stages___2___blocks___6___conv1_conv(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___6___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___6___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___6___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___6___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_192 = torch.nn.functional.batch_norm(x_191, getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_bias, False, 0.1, 1e-05);  x_191 = getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___6___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_193 = self.getattr_getattr_L__mod___stages___2___blocks___6___conv1_bn_drop(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_196 = self.getattr_getattr_L__mod___stages___2___blocks___6___conv1_bn_act(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_197 = self.getattr_getattr_L__mod___stages___2___blocks___6___attn(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_198 = self.getattr_getattr_L__mod___stages___2___blocks___6___conv2_conv(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___6___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___6___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___6___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___6___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_199 = torch.nn.functional.batch_norm(x_198, getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_bias, False, 0.1, 1e-05);  x_198 = getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___6___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_200 = self.getattr_getattr_L__mod___stages___2___blocks___6___conv2_bn_drop(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_203 = self.getattr_getattr_L__mod___stages___2___blocks___6___conv2_bn_act(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___6___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___6___drop_path(x_203);  x_203 = None
    shortcut_10 = getattr_getattr_l__mod___stages___2___blocks___6___drop_path + shortcut_9;  getattr_getattr_l__mod___stages___2___blocks___6___drop_path = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_205 = self.getattr_getattr_L__mod___stages___2___blocks___7___conv1_conv(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___7___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___7___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___7___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___7___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_206 = torch.nn.functional.batch_norm(x_205, getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_weight, getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_bias, False, 0.1, 1e-05);  x_205 = getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_weight = getattr_getattr_l__mod___stages___2___blocks___7___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_207 = self.getattr_getattr_L__mod___stages___2___blocks___7___conv1_bn_drop(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_210 = self.getattr_getattr_L__mod___stages___2___blocks___7___conv1_bn_act(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_211 = self.getattr_getattr_L__mod___stages___2___blocks___7___attn(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_212 = self.getattr_getattr_L__mod___stages___2___blocks___7___conv2_conv(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___2___blocks___7___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___2___blocks___7___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_weight = self.getattr_getattr_L__mod___stages___2___blocks___7___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_bias = self.getattr_getattr_L__mod___stages___2___blocks___7___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_213 = torch.nn.functional.batch_norm(x_212, getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_running_mean, getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_running_var, getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_weight, getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_bias, False, 0.1, 1e-05);  x_212 = getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_running_mean = getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_running_var = getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_weight = getattr_getattr_l__mod___stages___2___blocks___7___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_214 = self.getattr_getattr_L__mod___stages___2___blocks___7___conv2_bn_drop(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_217 = self.getattr_getattr_L__mod___stages___2___blocks___7___conv2_bn_act(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___2___blocks___7___drop_path = self.getattr_getattr_L__mod___stages___2___blocks___7___drop_path(x_217);  x_217 = None
    xb_7 = getattr_getattr_l__mod___stages___2___blocks___7___drop_path + shortcut_10;  getattr_getattr_l__mod___stages___2___blocks___7___drop_path = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_219 = self.getattr_L__mod___stages___2___conv_transition_b_conv(xb_7);  xb_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___2___conv_transition_b_bn_running_mean = self.getattr_L__mod___stages___2___conv_transition_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___2___conv_transition_b_bn_running_var = self.getattr_L__mod___stages___2___conv_transition_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___2___conv_transition_b_bn_weight = self.getattr_L__mod___stages___2___conv_transition_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___2___conv_transition_b_bn_bias = self.getattr_L__mod___stages___2___conv_transition_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_220 = torch.nn.functional.batch_norm(x_219, getattr_l__mod___stages___2___conv_transition_b_bn_running_mean, getattr_l__mod___stages___2___conv_transition_b_bn_running_var, getattr_l__mod___stages___2___conv_transition_b_bn_weight, getattr_l__mod___stages___2___conv_transition_b_bn_bias, False, 0.1, 1e-05);  x_219 = getattr_l__mod___stages___2___conv_transition_b_bn_running_mean = getattr_l__mod___stages___2___conv_transition_b_bn_running_var = getattr_l__mod___stages___2___conv_transition_b_bn_weight = getattr_l__mod___stages___2___conv_transition_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_221 = self.getattr_L__mod___stages___2___conv_transition_b_bn_drop(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_223 = self.getattr_L__mod___stages___2___conv_transition_b_bn_act(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:338, code: xb = self.conv_transition_b(xb).contiguous()
    xb_8 = x_223.contiguous();  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    cat_2 = torch.cat([xs_2, xb_8], dim = 1);  xs_2 = xb_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_224 = self.getattr_L__mod___stages___2___conv_transition_conv(cat_2);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___2___conv_transition_bn_running_mean = self.getattr_L__mod___stages___2___conv_transition_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___2___conv_transition_bn_running_var = self.getattr_L__mod___stages___2___conv_transition_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___2___conv_transition_bn_weight = self.getattr_L__mod___stages___2___conv_transition_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___2___conv_transition_bn_bias = self.getattr_L__mod___stages___2___conv_transition_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_225 = torch.nn.functional.batch_norm(x_224, getattr_l__mod___stages___2___conv_transition_bn_running_mean, getattr_l__mod___stages___2___conv_transition_bn_running_var, getattr_l__mod___stages___2___conv_transition_bn_weight, getattr_l__mod___stages___2___conv_transition_bn_bias, False, 0.1, 1e-05);  x_224 = getattr_l__mod___stages___2___conv_transition_bn_running_mean = getattr_l__mod___stages___2___conv_transition_bn_running_var = getattr_l__mod___stages___2___conv_transition_bn_weight = getattr_l__mod___stages___2___conv_transition_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_226 = self.getattr_L__mod___stages___2___conv_transition_bn_drop(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    out_2 = self.getattr_L__mod___stages___2___conv_transition_bn_act(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    x_229 = self.getattr_L__mod___stages___3___conv_down_conv(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___3___conv_down_bn_running_mean = self.getattr_L__mod___stages___3___conv_down_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___3___conv_down_bn_running_var = self.getattr_L__mod___stages___3___conv_down_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___3___conv_down_bn_weight = self.getattr_L__mod___stages___3___conv_down_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___3___conv_down_bn_bias = self.getattr_L__mod___stages___3___conv_down_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_230 = torch.nn.functional.batch_norm(x_229, getattr_l__mod___stages___3___conv_down_bn_running_mean, getattr_l__mod___stages___3___conv_down_bn_running_var, getattr_l__mod___stages___3___conv_down_bn_weight, getattr_l__mod___stages___3___conv_down_bn_bias, False, 0.1, 1e-05);  x_229 = getattr_l__mod___stages___3___conv_down_bn_running_mean = getattr_l__mod___stages___3___conv_down_bn_running_var = getattr_l__mod___stages___3___conv_down_bn_weight = getattr_l__mod___stages___3___conv_down_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_231 = self.getattr_L__mod___stages___3___conv_down_bn_drop(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_233 = self.getattr_L__mod___stages___3___conv_down_bn_act(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:128, code: x = self.aa(x)
    x_235 = self.getattr_L__mod___stages___3___conv_down_aa(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_236 = self.getattr_L__mod___stages___3___conv_exp_conv(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___3___conv_exp_bn_running_mean = self.getattr_L__mod___stages___3___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___3___conv_exp_bn_running_var = self.getattr_L__mod___stages___3___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___3___conv_exp_bn_weight = self.getattr_L__mod___stages___3___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___3___conv_exp_bn_bias = self.getattr_L__mod___stages___3___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_237 = torch.nn.functional.batch_norm(x_236, getattr_l__mod___stages___3___conv_exp_bn_running_mean, getattr_l__mod___stages___3___conv_exp_bn_running_var, getattr_l__mod___stages___3___conv_exp_bn_weight, getattr_l__mod___stages___3___conv_exp_bn_bias, False, 0.1, 1e-05);  x_236 = getattr_l__mod___stages___3___conv_exp_bn_running_mean = getattr_l__mod___stages___3___conv_exp_bn_running_var = getattr_l__mod___stages___3___conv_exp_bn_weight = getattr_l__mod___stages___3___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_238 = self.getattr_L__mod___stages___3___conv_exp_bn_drop(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_241 = self.getattr_L__mod___stages___3___conv_exp_bn_act(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    split_3 = x_241.split(256, dim = 1);  x_241 = None
    xs_3 = split_3[0]
    shortcut_11 = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_242 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv1_conv(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_243 = torch.nn.functional.batch_norm(x_242, getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_bias, False, 0.1, 1e-05);  x_242 = getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_244 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv1_bn_drop(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_247 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv1_bn_act(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_248 = self.getattr_getattr_L__mod___stages___3___blocks___0___attn(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_249 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv2_conv(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___0___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___0___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_250 = torch.nn.functional.batch_norm(x_249, getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_bias, False, 0.1, 1e-05);  x_249 = getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___0___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_251 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv2_bn_drop(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_254 = self.getattr_getattr_L__mod___stages___3___blocks___0___conv2_bn_act(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___0___drop_path(x_254);  x_254 = None
    shortcut_12 = getattr_getattr_l__mod___stages___3___blocks___0___drop_path + shortcut_11;  getattr_getattr_l__mod___stages___3___blocks___0___drop_path = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_256 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv1_conv(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_257 = torch.nn.functional.batch_norm(x_256, getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_bias, False, 0.1, 1e-05);  x_256 = getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_258 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv1_bn_drop(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_261 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv1_bn_act(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_262 = self.getattr_getattr_L__mod___stages___3___blocks___1___attn(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_263 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv2_conv(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___1___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___1___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_264 = torch.nn.functional.batch_norm(x_263, getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_bias, False, 0.1, 1e-05);  x_263 = getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___1___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_265 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv2_bn_drop(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_268 = self.getattr_getattr_L__mod___stages___3___blocks___1___conv2_bn_act(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___1___drop_path(x_268);  x_268 = None
    shortcut_13 = getattr_getattr_l__mod___stages___3___blocks___1___drop_path + shortcut_12;  getattr_getattr_l__mod___stages___3___blocks___1___drop_path = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_270 = self.getattr_getattr_L__mod___stages___3___blocks___2___conv1_conv(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___2___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___2___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___2___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___2___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_271 = torch.nn.functional.batch_norm(x_270, getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_bias, False, 0.1, 1e-05);  x_270 = getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___2___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_272 = self.getattr_getattr_L__mod___stages___3___blocks___2___conv1_bn_drop(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_275 = self.getattr_getattr_L__mod___stages___3___blocks___2___conv1_bn_act(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_276 = self.getattr_getattr_L__mod___stages___3___blocks___2___attn(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_277 = self.getattr_getattr_L__mod___stages___3___blocks___2___conv2_conv(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___2___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___2___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___2___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___2___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_278 = torch.nn.functional.batch_norm(x_277, getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_bias, False, 0.1, 1e-05);  x_277 = getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___2___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_279 = self.getattr_getattr_L__mod___stages___3___blocks___2___conv2_bn_drop(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_282 = self.getattr_getattr_L__mod___stages___3___blocks___2___conv2_bn_act(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___2___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___2___drop_path(x_282);  x_282 = None
    shortcut_14 = getattr_getattr_l__mod___stages___3___blocks___2___drop_path + shortcut_13;  getattr_getattr_l__mod___stages___3___blocks___2___drop_path = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_284 = self.getattr_getattr_L__mod___stages___3___blocks___3___conv1_conv(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___3___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___3___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___3___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___3___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_285 = torch.nn.functional.batch_norm(x_284, getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_bias, False, 0.1, 1e-05);  x_284 = getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___3___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_286 = self.getattr_getattr_L__mod___stages___3___blocks___3___conv1_bn_drop(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_289 = self.getattr_getattr_L__mod___stages___3___blocks___3___conv1_bn_act(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_290 = self.getattr_getattr_L__mod___stages___3___blocks___3___attn(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_291 = self.getattr_getattr_L__mod___stages___3___blocks___3___conv2_conv(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___3___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___3___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___3___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___3___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_292 = torch.nn.functional.batch_norm(x_291, getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_bias, False, 0.1, 1e-05);  x_291 = getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___3___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_293 = self.getattr_getattr_L__mod___stages___3___blocks___3___conv2_bn_drop(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_296 = self.getattr_getattr_L__mod___stages___3___blocks___3___conv2_bn_act(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___3___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___3___drop_path(x_296);  x_296 = None
    shortcut_15 = getattr_getattr_l__mod___stages___3___blocks___3___drop_path + shortcut_14;  getattr_getattr_l__mod___stages___3___blocks___3___drop_path = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_298 = self.getattr_getattr_L__mod___stages___3___blocks___4___conv1_conv(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___4___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___4___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___4___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___4___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_299 = torch.nn.functional.batch_norm(x_298, getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_bias, False, 0.1, 1e-05);  x_298 = getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___4___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_300 = self.getattr_getattr_L__mod___stages___3___blocks___4___conv1_bn_drop(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_303 = self.getattr_getattr_L__mod___stages___3___blocks___4___conv1_bn_act(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_304 = self.getattr_getattr_L__mod___stages___3___blocks___4___attn(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_305 = self.getattr_getattr_L__mod___stages___3___blocks___4___conv2_conv(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___4___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___4___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___4___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___4___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_306 = torch.nn.functional.batch_norm(x_305, getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_bias, False, 0.1, 1e-05);  x_305 = getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___4___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_307 = self.getattr_getattr_L__mod___stages___3___blocks___4___conv2_bn_drop(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_310 = self.getattr_getattr_L__mod___stages___3___blocks___4___conv2_bn_act(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___4___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___4___drop_path(x_310);  x_310 = None
    shortcut_16 = getattr_getattr_l__mod___stages___3___blocks___4___drop_path + shortcut_15;  getattr_getattr_l__mod___stages___3___blocks___4___drop_path = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_312 = self.getattr_getattr_L__mod___stages___3___blocks___5___conv1_conv(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___5___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___5___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___5___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___5___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_313 = torch.nn.functional.batch_norm(x_312, getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_bias, False, 0.1, 1e-05);  x_312 = getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___5___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_314 = self.getattr_getattr_L__mod___stages___3___blocks___5___conv1_bn_drop(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_317 = self.getattr_getattr_L__mod___stages___3___blocks___5___conv1_bn_act(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_318 = self.getattr_getattr_L__mod___stages___3___blocks___5___attn(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_319 = self.getattr_getattr_L__mod___stages___3___blocks___5___conv2_conv(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___5___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___5___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___5___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___5___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_320 = torch.nn.functional.batch_norm(x_319, getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_bias, False, 0.1, 1e-05);  x_319 = getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___5___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_321 = self.getattr_getattr_L__mod___stages___3___blocks___5___conv2_bn_drop(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_324 = self.getattr_getattr_L__mod___stages___3___blocks___5___conv2_bn_act(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___5___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___5___drop_path(x_324);  x_324 = None
    shortcut_17 = getattr_getattr_l__mod___stages___3___blocks___5___drop_path + shortcut_16;  getattr_getattr_l__mod___stages___3___blocks___5___drop_path = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_326 = self.getattr_getattr_L__mod___stages___3___blocks___6___conv1_conv(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___6___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___6___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___6___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___6___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_327 = torch.nn.functional.batch_norm(x_326, getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_bias, False, 0.1, 1e-05);  x_326 = getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___6___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_328 = self.getattr_getattr_L__mod___stages___3___blocks___6___conv1_bn_drop(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_331 = self.getattr_getattr_L__mod___stages___3___blocks___6___conv1_bn_act(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_332 = self.getattr_getattr_L__mod___stages___3___blocks___6___attn(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_333 = self.getattr_getattr_L__mod___stages___3___blocks___6___conv2_conv(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___6___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___6___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___6___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___6___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_334 = torch.nn.functional.batch_norm(x_333, getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_bias, False, 0.1, 1e-05);  x_333 = getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___6___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_335 = self.getattr_getattr_L__mod___stages___3___blocks___6___conv2_bn_drop(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_338 = self.getattr_getattr_L__mod___stages___3___blocks___6___conv2_bn_act(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___6___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___6___drop_path(x_338);  x_338 = None
    shortcut_18 = getattr_getattr_l__mod___stages___3___blocks___6___drop_path + shortcut_17;  getattr_getattr_l__mod___stages___3___blocks___6___drop_path = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_340 = self.getattr_getattr_L__mod___stages___3___blocks___7___conv1_conv(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___7___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___7___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___7___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___7___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_341 = torch.nn.functional.batch_norm(x_340, getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_weight, getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_bias, False, 0.1, 1e-05);  x_340 = getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_weight = getattr_getattr_l__mod___stages___3___blocks___7___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_342 = self.getattr_getattr_L__mod___stages___3___blocks___7___conv1_bn_drop(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_345 = self.getattr_getattr_L__mod___stages___3___blocks___7___conv1_bn_act(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_346 = self.getattr_getattr_L__mod___stages___3___blocks___7___attn(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_347 = self.getattr_getattr_L__mod___stages___3___blocks___7___conv2_conv(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___3___blocks___7___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___3___blocks___7___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_weight = self.getattr_getattr_L__mod___stages___3___blocks___7___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_bias = self.getattr_getattr_L__mod___stages___3___blocks___7___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_348 = torch.nn.functional.batch_norm(x_347, getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_running_mean, getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_running_var, getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_weight, getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_bias, False, 0.1, 1e-05);  x_347 = getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_running_mean = getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_running_var = getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_weight = getattr_getattr_l__mod___stages___3___blocks___7___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_349 = self.getattr_getattr_L__mod___stages___3___blocks___7___conv2_bn_drop(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_352 = self.getattr_getattr_L__mod___stages___3___blocks___7___conv2_bn_act(x_349);  x_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___3___blocks___7___drop_path = self.getattr_getattr_L__mod___stages___3___blocks___7___drop_path(x_352);  x_352 = None
    xb_10 = getattr_getattr_l__mod___stages___3___blocks___7___drop_path + shortcut_18;  getattr_getattr_l__mod___stages___3___blocks___7___drop_path = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_354 = self.getattr_L__mod___stages___3___conv_transition_b_conv(xb_10);  xb_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___3___conv_transition_b_bn_running_mean = self.getattr_L__mod___stages___3___conv_transition_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___3___conv_transition_b_bn_running_var = self.getattr_L__mod___stages___3___conv_transition_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___3___conv_transition_b_bn_weight = self.getattr_L__mod___stages___3___conv_transition_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___3___conv_transition_b_bn_bias = self.getattr_L__mod___stages___3___conv_transition_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_355 = torch.nn.functional.batch_norm(x_354, getattr_l__mod___stages___3___conv_transition_b_bn_running_mean, getattr_l__mod___stages___3___conv_transition_b_bn_running_var, getattr_l__mod___stages___3___conv_transition_b_bn_weight, getattr_l__mod___stages___3___conv_transition_b_bn_bias, False, 0.1, 1e-05);  x_354 = getattr_l__mod___stages___3___conv_transition_b_bn_running_mean = getattr_l__mod___stages___3___conv_transition_b_bn_running_var = getattr_l__mod___stages___3___conv_transition_b_bn_weight = getattr_l__mod___stages___3___conv_transition_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_356 = self.getattr_L__mod___stages___3___conv_transition_b_bn_drop(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_358 = self.getattr_L__mod___stages___3___conv_transition_b_bn_act(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:338, code: xb = self.conv_transition_b(xb).contiguous()
    xb_11 = x_358.contiguous();  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    cat_3 = torch.cat([xs_3, xb_11], dim = 1);  xs_3 = xb_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_359 = self.getattr_L__mod___stages___3___conv_transition_conv(cat_3);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___3___conv_transition_bn_running_mean = self.getattr_L__mod___stages___3___conv_transition_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___3___conv_transition_bn_running_var = self.getattr_L__mod___stages___3___conv_transition_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___3___conv_transition_bn_weight = self.getattr_L__mod___stages___3___conv_transition_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___3___conv_transition_bn_bias = self.getattr_L__mod___stages___3___conv_transition_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_360 = torch.nn.functional.batch_norm(x_359, getattr_l__mod___stages___3___conv_transition_bn_running_mean, getattr_l__mod___stages___3___conv_transition_bn_running_var, getattr_l__mod___stages___3___conv_transition_bn_weight, getattr_l__mod___stages___3___conv_transition_bn_bias, False, 0.1, 1e-05);  x_359 = getattr_l__mod___stages___3___conv_transition_bn_running_mean = getattr_l__mod___stages___3___conv_transition_bn_running_var = getattr_l__mod___stages___3___conv_transition_bn_weight = getattr_l__mod___stages___3___conv_transition_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_361 = self.getattr_L__mod___stages___3___conv_transition_bn_drop(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    out_3 = self.getattr_L__mod___stages___3___conv_transition_bn_act(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    x_364 = self.getattr_L__mod___stages___4___conv_down_conv(out_3);  out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___4___conv_down_bn_running_mean = self.getattr_L__mod___stages___4___conv_down_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___4___conv_down_bn_running_var = self.getattr_L__mod___stages___4___conv_down_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___4___conv_down_bn_weight = self.getattr_L__mod___stages___4___conv_down_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___4___conv_down_bn_bias = self.getattr_L__mod___stages___4___conv_down_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_365 = torch.nn.functional.batch_norm(x_364, getattr_l__mod___stages___4___conv_down_bn_running_mean, getattr_l__mod___stages___4___conv_down_bn_running_var, getattr_l__mod___stages___4___conv_down_bn_weight, getattr_l__mod___stages___4___conv_down_bn_bias, False, 0.1, 1e-05);  x_364 = getattr_l__mod___stages___4___conv_down_bn_running_mean = getattr_l__mod___stages___4___conv_down_bn_running_var = getattr_l__mod___stages___4___conv_down_bn_weight = getattr_l__mod___stages___4___conv_down_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_366 = self.getattr_L__mod___stages___4___conv_down_bn_drop(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_368 = self.getattr_L__mod___stages___4___conv_down_bn_act(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:128, code: x = self.aa(x)
    x_370 = self.getattr_L__mod___stages___4___conv_down_aa(x_368);  x_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_371 = self.getattr_L__mod___stages___4___conv_exp_conv(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___4___conv_exp_bn_running_mean = self.getattr_L__mod___stages___4___conv_exp_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___4___conv_exp_bn_running_var = self.getattr_L__mod___stages___4___conv_exp_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___4___conv_exp_bn_weight = self.getattr_L__mod___stages___4___conv_exp_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___4___conv_exp_bn_bias = self.getattr_L__mod___stages___4___conv_exp_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_372 = torch.nn.functional.batch_norm(x_371, getattr_l__mod___stages___4___conv_exp_bn_running_mean, getattr_l__mod___stages___4___conv_exp_bn_running_var, getattr_l__mod___stages___4___conv_exp_bn_weight, getattr_l__mod___stages___4___conv_exp_bn_bias, False, 0.1, 1e-05);  x_371 = getattr_l__mod___stages___4___conv_exp_bn_running_mean = getattr_l__mod___stages___4___conv_exp_bn_running_var = getattr_l__mod___stages___4___conv_exp_bn_weight = getattr_l__mod___stages___4___conv_exp_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_373 = self.getattr_L__mod___stages___4___conv_exp_bn_drop(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_376 = self.getattr_L__mod___stages___4___conv_exp_bn_act(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    split_4 = x_376.split(512, dim = 1);  x_376 = None
    xs_4 = split_4[0]
    shortcut_19 = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_377 = self.getattr_getattr_L__mod___stages___4___blocks___0___conv1_conv(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___0___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___0___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___0___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___0___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_378 = torch.nn.functional.batch_norm(x_377, getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_weight, getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_bias, False, 0.1, 1e-05);  x_377 = getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_weight = getattr_getattr_l__mod___stages___4___blocks___0___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_379 = self.getattr_getattr_L__mod___stages___4___blocks___0___conv1_bn_drop(x_378);  x_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_382 = self.getattr_getattr_L__mod___stages___4___blocks___0___conv1_bn_act(x_379);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_383 = self.getattr_getattr_L__mod___stages___4___blocks___0___attn(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_384 = self.getattr_getattr_L__mod___stages___4___blocks___0___conv2_conv(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___0___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___0___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___0___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___0___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_385 = torch.nn.functional.batch_norm(x_384, getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_weight, getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_bias, False, 0.1, 1e-05);  x_384 = getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_weight = getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_386 = self.getattr_getattr_L__mod___stages___4___blocks___0___conv2_bn_drop(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_389 = self.getattr_getattr_L__mod___stages___4___blocks___0___conv2_bn_act(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___4___blocks___0___drop_path = self.getattr_getattr_L__mod___stages___4___blocks___0___drop_path(x_389);  x_389 = None
    shortcut_20 = getattr_getattr_l__mod___stages___4___blocks___0___drop_path + shortcut_19;  getattr_getattr_l__mod___stages___4___blocks___0___drop_path = shortcut_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_391 = self.getattr_getattr_L__mod___stages___4___blocks___1___conv1_conv(shortcut_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___1___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___1___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___1___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___1___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_392 = torch.nn.functional.batch_norm(x_391, getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_weight, getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_bias, False, 0.1, 1e-05);  x_391 = getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_weight = getattr_getattr_l__mod___stages___4___blocks___1___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_393 = self.getattr_getattr_L__mod___stages___4___blocks___1___conv1_bn_drop(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_396 = self.getattr_getattr_L__mod___stages___4___blocks___1___conv1_bn_act(x_393);  x_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_397 = self.getattr_getattr_L__mod___stages___4___blocks___1___attn(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_398 = self.getattr_getattr_L__mod___stages___4___blocks___1___conv2_conv(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___1___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___1___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___1___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___1___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_399 = torch.nn.functional.batch_norm(x_398, getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_weight, getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_bias, False, 0.1, 1e-05);  x_398 = getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_weight = getattr_getattr_l__mod___stages___4___blocks___1___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_400 = self.getattr_getattr_L__mod___stages___4___blocks___1___conv2_bn_drop(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_403 = self.getattr_getattr_L__mod___stages___4___blocks___1___conv2_bn_act(x_400);  x_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___4___blocks___1___drop_path = self.getattr_getattr_L__mod___stages___4___blocks___1___drop_path(x_403);  x_403 = None
    shortcut_21 = getattr_getattr_l__mod___stages___4___blocks___1___drop_path + shortcut_20;  getattr_getattr_l__mod___stages___4___blocks___1___drop_path = shortcut_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_405 = self.getattr_getattr_L__mod___stages___4___blocks___2___conv1_conv(shortcut_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___2___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___2___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___2___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___2___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_406 = torch.nn.functional.batch_norm(x_405, getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_weight, getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_bias, False, 0.1, 1e-05);  x_405 = getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_weight = getattr_getattr_l__mod___stages___4___blocks___2___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_407 = self.getattr_getattr_L__mod___stages___4___blocks___2___conv1_bn_drop(x_406);  x_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_410 = self.getattr_getattr_L__mod___stages___4___blocks___2___conv1_bn_act(x_407);  x_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_411 = self.getattr_getattr_L__mod___stages___4___blocks___2___attn(x_410);  x_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_412 = self.getattr_getattr_L__mod___stages___4___blocks___2___conv2_conv(x_411);  x_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___2___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___2___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___2___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___2___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_413 = torch.nn.functional.batch_norm(x_412, getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_weight, getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_bias, False, 0.1, 1e-05);  x_412 = getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_weight = getattr_getattr_l__mod___stages___4___blocks___2___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_414 = self.getattr_getattr_L__mod___stages___4___blocks___2___conv2_bn_drop(x_413);  x_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_417 = self.getattr_getattr_L__mod___stages___4___blocks___2___conv2_bn_act(x_414);  x_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___4___blocks___2___drop_path = self.getattr_getattr_L__mod___stages___4___blocks___2___drop_path(x_417);  x_417 = None
    shortcut_22 = getattr_getattr_l__mod___stages___4___blocks___2___drop_path + shortcut_21;  getattr_getattr_l__mod___stages___4___blocks___2___drop_path = shortcut_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_419 = self.getattr_getattr_L__mod___stages___4___blocks___3___conv1_conv(shortcut_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___3___conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___3___conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___3___conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___3___conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_420 = torch.nn.functional.batch_norm(x_419, getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_weight, getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_bias, False, 0.1, 1e-05);  x_419 = getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_weight = getattr_getattr_l__mod___stages___4___blocks___3___conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_421 = self.getattr_getattr_L__mod___stages___4___blocks___3___conv1_bn_drop(x_420);  x_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_424 = self.getattr_getattr_L__mod___stages___4___blocks___3___conv1_bn_act(x_421);  x_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:220, code: x = self.attn(x)
    x_425 = self.getattr_getattr_L__mod___stages___4___blocks___3___attn(x_424);  x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_426 = self.getattr_getattr_L__mod___stages___4___blocks___3___conv2_conv(x_425);  x_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_running_mean = self.getattr_getattr_L__mod___stages___4___blocks___3___conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_running_var = self.getattr_getattr_L__mod___stages___4___blocks___3___conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_weight = self.getattr_getattr_L__mod___stages___4___blocks___3___conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_bias = self.getattr_getattr_L__mod___stages___4___blocks___3___conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_427 = torch.nn.functional.batch_norm(x_426, getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_running_mean, getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_running_var, getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_weight, getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_bias, False, 0.1, 1e-05);  x_426 = getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_running_mean = getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_running_var = getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_weight = getattr_getattr_l__mod___stages___4___blocks___3___conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_428 = self.getattr_getattr_L__mod___stages___4___blocks___3___conv2_bn_drop(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_431 = self.getattr_getattr_L__mod___stages___4___blocks___3___conv2_bn_act(x_428);  x_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___stages___4___blocks___3___drop_path = self.getattr_getattr_L__mod___stages___4___blocks___3___drop_path(x_431);  x_431 = None
    xb_13 = getattr_getattr_l__mod___stages___4___blocks___3___drop_path + shortcut_22;  getattr_getattr_l__mod___stages___4___blocks___3___drop_path = shortcut_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_433 = self.getattr_L__mod___stages___4___conv_transition_b_conv(xb_13);  xb_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___4___conv_transition_b_bn_running_mean = self.getattr_L__mod___stages___4___conv_transition_b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___4___conv_transition_b_bn_running_var = self.getattr_L__mod___stages___4___conv_transition_b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___4___conv_transition_b_bn_weight = self.getattr_L__mod___stages___4___conv_transition_b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___4___conv_transition_b_bn_bias = self.getattr_L__mod___stages___4___conv_transition_b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_434 = torch.nn.functional.batch_norm(x_433, getattr_l__mod___stages___4___conv_transition_b_bn_running_mean, getattr_l__mod___stages___4___conv_transition_b_bn_running_var, getattr_l__mod___stages___4___conv_transition_b_bn_weight, getattr_l__mod___stages___4___conv_transition_b_bn_bias, False, 0.1, 1e-05);  x_433 = getattr_l__mod___stages___4___conv_transition_b_bn_running_mean = getattr_l__mod___stages___4___conv_transition_b_bn_running_var = getattr_l__mod___stages___4___conv_transition_b_bn_weight = getattr_l__mod___stages___4___conv_transition_b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_435 = self.getattr_L__mod___stages___4___conv_transition_b_bn_drop(x_434);  x_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_437 = self.getattr_L__mod___stages___4___conv_transition_b_bn_act(x_435);  x_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:338, code: xb = self.conv_transition_b(xb).contiguous()
    xb_14 = x_437.contiguous();  x_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    cat_4 = torch.cat([xs_4, xb_14], dim = 1);  xs_4 = xb_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_438 = self.getattr_L__mod___stages___4___conv_transition_conv(cat_4);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_l__mod___stages___4___conv_transition_bn_running_mean = self.getattr_L__mod___stages___4___conv_transition_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_l__mod___stages___4___conv_transition_bn_running_var = self.getattr_L__mod___stages___4___conv_transition_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_l__mod___stages___4___conv_transition_bn_weight = self.getattr_L__mod___stages___4___conv_transition_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_l__mod___stages___4___conv_transition_bn_bias = self.getattr_L__mod___stages___4___conv_transition_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_439 = torch.nn.functional.batch_norm(x_438, getattr_l__mod___stages___4___conv_transition_bn_running_mean, getattr_l__mod___stages___4___conv_transition_bn_running_var, getattr_l__mod___stages___4___conv_transition_bn_weight, getattr_l__mod___stages___4___conv_transition_bn_bias, False, 0.1, 1e-05);  x_438 = getattr_l__mod___stages___4___conv_transition_bn_running_mean = getattr_l__mod___stages___4___conv_transition_bn_running_var = getattr_l__mod___stages___4___conv_transition_bn_weight = getattr_l__mod___stages___4___conv_transition_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_440 = self.getattr_L__mod___stages___4___conv_transition_bn_drop(x_439);  x_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_444 = self.getattr_L__mod___stages___4___conv_transition_bn_act(x_440);  x_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_445 = self.L__mod___head_global_pool_pool(x_444);  x_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_447 = self.L__mod___head_global_pool_flatten(x_445);  x_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_448 = self.L__mod___head_drop(x_447);  x_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_449 = self.L__mod___head_fc(x_448);  x_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_450 = self.L__mod___head_flatten(x_449);  x_449 = None
    return (x_450,)
    