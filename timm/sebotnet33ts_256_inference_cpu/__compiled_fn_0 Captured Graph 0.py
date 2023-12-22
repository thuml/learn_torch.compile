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
    x_4 = self.L__mod___stem_conv1_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_5 = self.L__mod___stem_conv2_conv(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___stem_conv2_bn_running_mean = self.L__mod___stem_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv2_bn_running_var = self.L__mod___stem_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv2_bn_weight = self.L__mod___stem_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv2_bn_bias = self.L__mod___stem_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_6 = torch.nn.functional.batch_norm(x_5, l__mod___stem_conv2_bn_running_mean, l__mod___stem_conv2_bn_running_var, l__mod___stem_conv2_bn_weight, l__mod___stem_conv2_bn_bias, False, 0.1, 1e-05);  x_5 = l__mod___stem_conv2_bn_running_mean = l__mod___stem_conv2_bn_running_var = l__mod___stem_conv2_bn_weight = l__mod___stem_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_7 = self.L__mod___stem_conv2_bn_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_9 = self.L__mod___stem_conv2_bn_act(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_10 = self.L__mod___stem_conv3_conv(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___stem_conv3_bn_running_mean = self.L__mod___stem_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv3_bn_running_var = self.L__mod___stem_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv3_bn_weight = self.L__mod___stem_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv3_bn_bias = self.L__mod___stem_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_11 = torch.nn.functional.batch_norm(x_10, l__mod___stem_conv3_bn_running_mean, l__mod___stem_conv3_bn_running_var, l__mod___stem_conv3_bn_weight, l__mod___stem_conv3_bn_bias, False, 0.1, 1e-05);  x_10 = l__mod___stem_conv3_bn_running_mean = l__mod___stem_conv3_bn_running_var = l__mod___stem_conv3_bn_weight = l__mod___stem_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_12 = self.L__mod___stem_conv3_bn_drop(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___stem_conv3_bn_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_16 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_conv(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_17 = torch.nn.functional.batch_norm(x_16, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_16 = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_18 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_21 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_act(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_22 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_conv(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_23 = torch.nn.functional.batch_norm(x_22, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_22 = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_24 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_drop(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_27 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_act(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_28 = self.getattr_getattr_L__mod___stages___0_____0___conv2b_kxk(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_28.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_1 = self.getattr_getattr_L__mod___stages___0_____0___attn_fc1(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___0_____0___attn_bn = self.getattr_getattr_L__mod___stages___0_____0___attn_bn(x_se_1);  x_se_1 = None
    x_se_2 = self.getattr_getattr_L__mod___stages___0_____0___attn_act(getattr_getattr_l__mod___stages___0_____0___attn_bn);  getattr_getattr_l__mod___stages___0_____0___attn_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_3 = self.getattr_getattr_L__mod___stages___0_____0___attn_fc2(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid = x_se_3.sigmoid();  x_se_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_29 = x_28 * sigmoid;  x_28 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_30 = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_conv(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_31 = torch.nn.functional.batch_norm(x_30, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_30 = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_32 = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_drop(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_35 = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_act(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_36 = self.getattr_getattr_L__mod___stages___0_____0___attn_last(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_37 = self.getattr_getattr_L__mod___stages___0_____0___drop_path(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_38 = self.getattr_getattr_L__mod___stages___0_____0___shortcut_conv(shortcut);  shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_39 = torch.nn.functional.batch_norm(x_38, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_38 = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_40 = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_drop(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_42 = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    x_43 = x_37 + x_42;  x_37 = x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_1 = self.getattr_getattr_L__mod___stages___0_____0___act(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_44 = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_conv(shortcut_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_45 = torch.nn.functional.batch_norm(x_44, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_44 = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_46 = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_drop(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_49 = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_act(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_50 = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_conv(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_51 = torch.nn.functional.batch_norm(x_50, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_50 = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_52 = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_55 = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_act(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_56 = self.getattr_getattr_L__mod___stages___0_____1___conv2b_kxk(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_56.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_5 = self.getattr_getattr_L__mod___stages___0_____1___attn_fc1(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___0_____1___attn_bn = self.getattr_getattr_L__mod___stages___0_____1___attn_bn(x_se_5);  x_se_5 = None
    x_se_6 = self.getattr_getattr_L__mod___stages___0_____1___attn_act(getattr_getattr_l__mod___stages___0_____1___attn_bn);  getattr_getattr_l__mod___stages___0_____1___attn_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_7 = self.getattr_getattr_L__mod___stages___0_____1___attn_fc2(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1 = x_se_7.sigmoid();  x_se_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_57 = x_56 * sigmoid_1;  x_56 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_58 = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_conv(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_59 = torch.nn.functional.batch_norm(x_58, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_58 = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_60 = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_drop(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_63 = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_act(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_64 = self.getattr_getattr_L__mod___stages___0_____1___attn_last(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_65 = self.getattr_getattr_L__mod___stages___0_____1___drop_path(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___0_____1___shortcut = self.getattr_getattr_L__mod___stages___0_____1___shortcut(shortcut_1);  shortcut_1 = None
    x_66 = x_65 + getattr_getattr_l__mod___stages___0_____1___shortcut;  x_65 = getattr_getattr_l__mod___stages___0_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___stages___0_____1___act(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_67 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_conv(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_68 = torch.nn.functional.batch_norm(x_67, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_67 = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_69 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_drop(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_72 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_73 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_conv(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_74 = torch.nn.functional.batch_norm(x_73, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_73 = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_75 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_drop(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_78 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_act(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_79 = self.getattr_getattr_L__mod___stages___1_____0___conv2b_kxk(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_79.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_9 = self.getattr_getattr_L__mod___stages___1_____0___attn_fc1(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___1_____0___attn_bn = self.getattr_getattr_L__mod___stages___1_____0___attn_bn(x_se_9);  x_se_9 = None
    x_se_10 = self.getattr_getattr_L__mod___stages___1_____0___attn_act(getattr_getattr_l__mod___stages___1_____0___attn_bn);  getattr_getattr_l__mod___stages___1_____0___attn_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_11 = self.getattr_getattr_L__mod___stages___1_____0___attn_fc2(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2 = x_se_11.sigmoid();  x_se_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_80 = x_79 * sigmoid_2;  x_79 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_81 = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_conv(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_82 = torch.nn.functional.batch_norm(x_81, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_81 = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_83 = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_drop(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_86 = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_act(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_87 = self.getattr_getattr_L__mod___stages___1_____0___attn_last(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_88 = self.getattr_getattr_L__mod___stages___1_____0___drop_path(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_89 = self.getattr_getattr_L__mod___stages___1_____0___shortcut_conv(shortcut_2);  shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_90 = torch.nn.functional.batch_norm(x_89, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_89 = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_91 = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_drop(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_93 = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_act(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    x_94 = x_88 + x_93;  x_88 = x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_3 = self.getattr_getattr_L__mod___stages___1_____0___act(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_95 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_conv(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_96 = torch.nn.functional.batch_norm(x_95, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_95 = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_97 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_drop(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_100 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_act(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_101 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_conv(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_102 = torch.nn.functional.batch_norm(x_101, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_101 = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_103 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_drop(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_106 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_act(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_107 = self.getattr_getattr_L__mod___stages___1_____1___conv2b_kxk(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_107.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_13 = self.getattr_getattr_L__mod___stages___1_____1___attn_fc1(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___1_____1___attn_bn = self.getattr_getattr_L__mod___stages___1_____1___attn_bn(x_se_13);  x_se_13 = None
    x_se_14 = self.getattr_getattr_L__mod___stages___1_____1___attn_act(getattr_getattr_l__mod___stages___1_____1___attn_bn);  getattr_getattr_l__mod___stages___1_____1___attn_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_15 = self.getattr_getattr_L__mod___stages___1_____1___attn_fc2(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3 = x_se_15.sigmoid();  x_se_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_108 = x_107 * sigmoid_3;  x_107 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_109 = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_conv(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_110 = torch.nn.functional.batch_norm(x_109, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_109 = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_111 = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_drop(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_114 = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_act(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_115 = self.getattr_getattr_L__mod___stages___1_____1___attn_last(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_116 = self.getattr_getattr_L__mod___stages___1_____1___drop_path(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1_____1___shortcut = self.getattr_getattr_L__mod___stages___1_____1___shortcut(shortcut_3);  shortcut_3 = None
    x_117 = x_116 + getattr_getattr_l__mod___stages___1_____1___shortcut;  x_116 = getattr_getattr_l__mod___stages___1_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___stages___1_____1___act(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_118 = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_conv(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_119 = torch.nn.functional.batch_norm(x_118, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_118 = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_120 = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_123 = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:882, code: x = self.conv2_kxk(x)
    x_124 = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    x_125 = self.getattr_getattr_L__mod___stages___1_____2___self_attn_qkv(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split = torch.functional.split(x_125, [128, 128, 128], dim = 1);  x_125 = None
    q = split[0]
    k = split[1]
    v = split[2];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    reshape = q.reshape(32, 32, -1);  q = None
    q_1 = reshape.transpose(-1, -2);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    k_1 = k.reshape(32, 32, -1);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    reshape_2 = v.reshape(32, 32, -1);  v = None
    v_1 = reshape_2.transpose(-1, -2);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    matmul = q_1 @ k_1;  k_1 = None
    mul_4 = matmul * 0.1767766952966369;  matmul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    q_2 = q_1.reshape(32, 32, 32, -1);  q_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:73, code: rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
    getattr_getattr_l__mod___stages___1_____2___self_attn_pos_embed_width_rel = self.getattr_getattr_L__mod___stages___1_____2___self_attn_pos_embed_width_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_2 = getattr_getattr_l__mod___stages___1_____2___self_attn_pos_embed_width_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___1_____2___self_attn_pos_embed_width_rel = None
    x_126 = q_2 @ transpose_2;  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_127 = x_126.reshape(-1, 32, 63);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad = torch.nn.functional.pad(x_127, [0, 1]);  x_127 = None
    x_pad = pad.flatten(1);  pad = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_1 = torch.nn.functional.pad(x_pad, [0, 31]);  x_pad = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_2 = x_pad_1.reshape(-1, 33, 63);  x_pad_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_128 = x_pad_2[(slice(None, None, None), slice(None, 32, None), slice(31, None, None))];  x_pad_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_6 = x_128.reshape(32, 32, 1, 32, 32);  x_128 = None
    x_129 = reshape_6.expand(-1, -1, 32, -1, -1);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_w = x_129.permute((0, 1, 3, 2, 4));  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    q_3 = q_2.transpose(1, 2);  q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:77, code: rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
    getattr_getattr_l__mod___stages___1_____2___self_attn_pos_embed_height_rel = self.getattr_getattr_L__mod___stages___1_____2___self_attn_pos_embed_height_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_4 = getattr_getattr_l__mod___stages___1_____2___self_attn_pos_embed_height_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___1_____2___self_attn_pos_embed_height_rel = None
    x_130 = q_3 @ transpose_4;  q_3 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_131 = x_130.reshape(-1, 32, 63);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_2 = torch.nn.functional.pad(x_131, [0, 1]);  x_131 = None
    x_pad_3 = pad_2.flatten(1);  pad_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_4 = torch.nn.functional.pad(x_pad_3, [0, 31]);  x_pad_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_5 = x_pad_4.reshape(-1, 33, 63);  x_pad_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_132 = x_pad_5[(slice(None, None, None), slice(None, 32, None), slice(31, None, None))];  x_pad_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_9 = x_132.reshape(32, 32, 1, 32, 32);  x_132 = None
    x_133 = reshape_9.expand(-1, -1, 32, -1, -1);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_h = x_133.permute((0, 3, 1, 4, 2));  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    rel_logits = rel_logits_h + rel_logits_w;  rel_logits_h = rel_logits_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    rel_logits_1 = rel_logits.reshape(32, 1024, 1024);  rel_logits = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    attn = mul_4 + rel_logits_1;  mul_4 = rel_logits_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    attn_1 = attn.softmax(dim = -1);  attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    matmul_3 = attn_1 @ v_1;  attn_1 = v_1 = None
    transpose_5 = matmul_3.transpose(-1, -2);  matmul_3 = None
    out = transpose_5.reshape(8, 128, 32, 32);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    x_134 = self.getattr_getattr_L__mod___stages___1_____2___self_attn_pool(out);  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____2___post_attn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___post_attn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___post_attn_running_var = self.getattr_getattr_L__mod___stages___1_____2___post_attn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___post_attn_weight = self.getattr_getattr_L__mod___stages___1_____2___post_attn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___post_attn_bias = self.getattr_getattr_L__mod___stages___1_____2___post_attn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_135 = torch.nn.functional.batch_norm(x_134, getattr_getattr_l__mod___stages___1_____2___post_attn_running_mean, getattr_getattr_l__mod___stages___1_____2___post_attn_running_var, getattr_getattr_l__mod___stages___1_____2___post_attn_weight, getattr_getattr_l__mod___stages___1_____2___post_attn_bias, False, 0.1, 1e-05);  x_134 = getattr_getattr_l__mod___stages___1_____2___post_attn_running_mean = getattr_getattr_l__mod___stages___1_____2___post_attn_running_var = getattr_getattr_l__mod___stages___1_____2___post_attn_weight = getattr_getattr_l__mod___stages___1_____2___post_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_136 = self.getattr_getattr_L__mod___stages___1_____2___post_attn_drop(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_138 = self.getattr_getattr_L__mod___stages___1_____2___post_attn_act(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_139 = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_conv(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_140 = torch.nn.functional.batch_norm(x_139, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_139 = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_141 = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_drop(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_144 = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_act(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:886, code: x = self.drop_path(x)
    x_145 = self.getattr_getattr_L__mod___stages___1_____2___drop_path(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1_____2___shortcut = self.getattr_getattr_L__mod___stages___1_____2___shortcut(shortcut_4);  shortcut_4 = None
    x_146 = x_145 + getattr_getattr_l__mod___stages___1_____2___shortcut;  x_145 = getattr_getattr_l__mod___stages___1_____2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    shortcut_5 = self.getattr_getattr_L__mod___stages___1_____2___act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_147 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_conv(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_148 = torch.nn.functional.batch_norm(x_147, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_147 = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_149 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_drop(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_152 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_act(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_153 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_conv(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_154 = torch.nn.functional.batch_norm(x_153, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_153 = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_155 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_drop(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_158 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_act(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_159 = self.getattr_getattr_L__mod___stages___2_____0___conv2b_kxk(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_159.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_17 = self.getattr_getattr_L__mod___stages___2_____0___attn_fc1(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____0___attn_bn = self.getattr_getattr_L__mod___stages___2_____0___attn_bn(x_se_17);  x_se_17 = None
    x_se_18 = self.getattr_getattr_L__mod___stages___2_____0___attn_act(getattr_getattr_l__mod___stages___2_____0___attn_bn);  getattr_getattr_l__mod___stages___2_____0___attn_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_19 = self.getattr_getattr_L__mod___stages___2_____0___attn_fc2(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4 = x_se_19.sigmoid();  x_se_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_160 = x_159 * sigmoid_4;  x_159 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_161 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_conv(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_162 = torch.nn.functional.batch_norm(x_161, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_161 = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_163 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_drop(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_166 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_act(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_167 = self.getattr_getattr_L__mod___stages___2_____0___attn_last(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_168 = self.getattr_getattr_L__mod___stages___2_____0___drop_path(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_169 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_conv(shortcut_5);  shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_170 = torch.nn.functional.batch_norm(x_169, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_169 = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_171 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_drop(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_173 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_act(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    x_174 = x_168 + x_173;  x_168 = x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_6 = self.getattr_getattr_L__mod___stages___2_____0___act(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_175 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_conv(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_176 = torch.nn.functional.batch_norm(x_175, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_175 = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_177 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_drop(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_180 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_act(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_181 = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_conv(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_182 = torch.nn.functional.batch_norm(x_181, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_bias, False, 0.1, 1e-05);  x_181 = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_183 = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_drop(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_186 = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk_bn_act(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_187 = self.getattr_getattr_L__mod___stages___2_____1___conv2b_kxk(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_187.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    x_se_21 = self.getattr_getattr_L__mod___stages___2_____1___attn_fc1(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    getattr_getattr_l__mod___stages___2_____1___attn_bn = self.getattr_getattr_L__mod___stages___2_____1___attn_bn(x_se_21);  x_se_21 = None
    x_se_22 = self.getattr_getattr_L__mod___stages___2_____1___attn_act(getattr_getattr_l__mod___stages___2_____1___attn_bn);  getattr_getattr_l__mod___stages___2_____1___attn_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    x_se_23 = self.getattr_getattr_L__mod___stages___2_____1___attn_fc2(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5 = x_se_23.sigmoid();  x_se_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    x_188 = x_187 * sigmoid_5;  x_187 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_189 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_conv(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_190 = torch.nn.functional.batch_norm(x_189, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_189 = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_191 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_drop(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_194 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_act(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_195 = self.getattr_getattr_L__mod___stages___2_____1___attn_last(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_196 = self.getattr_getattr_L__mod___stages___2_____1___drop_path(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____1___shortcut = self.getattr_getattr_L__mod___stages___2_____1___shortcut(shortcut_6);  shortcut_6 = None
    x_197 = x_196 + getattr_getattr_l__mod___stages___2_____1___shortcut;  x_196 = getattr_getattr_l__mod___stages___2_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_7 = self.getattr_getattr_L__mod___stages___2_____1___act(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_198 = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_199 = torch.nn.functional.batch_norm(x_198, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_198 = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____2___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_200 = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_drop(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_203 = self.getattr_getattr_L__mod___stages___2_____2___conv1_1x1_bn_act(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:882, code: x = self.conv2_kxk(x)
    x_204 = self.getattr_getattr_L__mod___stages___2_____2___conv2_kxk(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    x_205 = self.getattr_getattr_L__mod___stages___2_____2___self_attn_qkv(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_1 = torch.functional.split(x_205, [256, 256, 256], dim = 1);  x_205 = None
    q_4 = split_1[0]
    k_2 = split_1[1]
    v_2 = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    reshape_12 = q_4.reshape(32, 64, -1);  q_4 = None
    q_5 = reshape_12.transpose(-1, -2);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    k_3 = k_2.reshape(32, 64, -1);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    reshape_14 = v_2.reshape(32, 64, -1);  v_2 = None
    v_3 = reshape_14.transpose(-1, -2);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    matmul_4 = q_5 @ k_3;  k_3 = None
    mul_7 = matmul_4 * 0.125;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    q_6 = q_5.reshape(32, 16, 16, -1);  q_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:73, code: rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
    getattr_getattr_l__mod___stages___2_____2___self_attn_pos_embed_width_rel = self.getattr_getattr_L__mod___stages___2_____2___self_attn_pos_embed_width_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_8 = getattr_getattr_l__mod___stages___2_____2___self_attn_pos_embed_width_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___2_____2___self_attn_pos_embed_width_rel = None
    x_206 = q_6 @ transpose_8;  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_207 = x_206.reshape(-1, 16, 31);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_4 = torch.nn.functional.pad(x_207, [0, 1]);  x_207 = None
    x_pad_6 = pad_4.flatten(1);  pad_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_7 = torch.nn.functional.pad(x_pad_6, [0, 15]);  x_pad_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_8 = x_pad_7.reshape(-1, 17, 31);  x_pad_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_208 = x_pad_8[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))];  x_pad_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_18 = x_208.reshape(32, 16, 1, 16, 16);  x_208 = None
    x_209 = reshape_18.expand(-1, -1, 16, -1, -1);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_w_1 = x_209.permute((0, 1, 3, 2, 4));  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    q_7 = q_6.transpose(1, 2);  q_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:77, code: rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
    getattr_getattr_l__mod___stages___2_____2___self_attn_pos_embed_height_rel = self.getattr_getattr_L__mod___stages___2_____2___self_attn_pos_embed_height_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_10 = getattr_getattr_l__mod___stages___2_____2___self_attn_pos_embed_height_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___2_____2___self_attn_pos_embed_height_rel = None
    x_210 = q_7 @ transpose_10;  q_7 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_211 = x_210.reshape(-1, 16, 31);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_6 = torch.nn.functional.pad(x_211, [0, 1]);  x_211 = None
    x_pad_9 = pad_6.flatten(1);  pad_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_10 = torch.nn.functional.pad(x_pad_9, [0, 15]);  x_pad_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_11 = x_pad_10.reshape(-1, 17, 31);  x_pad_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_212 = x_pad_11[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))];  x_pad_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_21 = x_212.reshape(32, 16, 1, 16, 16);  x_212 = None
    x_213 = reshape_21.expand(-1, -1, 16, -1, -1);  reshape_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_h_1 = x_213.permute((0, 3, 1, 4, 2));  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    rel_logits_2 = rel_logits_h_1 + rel_logits_w_1;  rel_logits_h_1 = rel_logits_w_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    rel_logits_3 = rel_logits_2.reshape(32, 256, 256);  rel_logits_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    attn_2 = mul_7 + rel_logits_3;  mul_7 = rel_logits_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    attn_3 = attn_2.softmax(dim = -1);  attn_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    matmul_7 = attn_3 @ v_3;  attn_3 = v_3 = None
    transpose_11 = matmul_7.transpose(-1, -2);  matmul_7 = None
    out_2 = transpose_11.reshape(8, 256, 16, 16);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    x_214 = self.getattr_getattr_L__mod___stages___2_____2___self_attn_pool(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___post_attn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___post_attn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___post_attn_running_var = self.getattr_getattr_L__mod___stages___2_____2___post_attn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___post_attn_weight = self.getattr_getattr_L__mod___stages___2_____2___post_attn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___post_attn_bias = self.getattr_getattr_L__mod___stages___2_____2___post_attn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_215 = torch.nn.functional.batch_norm(x_214, getattr_getattr_l__mod___stages___2_____2___post_attn_running_mean, getattr_getattr_l__mod___stages___2_____2___post_attn_running_var, getattr_getattr_l__mod___stages___2_____2___post_attn_weight, getattr_getattr_l__mod___stages___2_____2___post_attn_bias, False, 0.1, 1e-05);  x_214 = getattr_getattr_l__mod___stages___2_____2___post_attn_running_mean = getattr_getattr_l__mod___stages___2_____2___post_attn_running_var = getattr_getattr_l__mod___stages___2_____2___post_attn_weight = getattr_getattr_l__mod___stages___2_____2___post_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_216 = self.getattr_getattr_L__mod___stages___2_____2___post_attn_drop(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_218 = self.getattr_getattr_L__mod___stages___2_____2___post_attn_act(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_219 = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_conv(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_220 = torch.nn.functional.batch_norm(x_219, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_219 = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____2___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_221 = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_drop(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_224 = self.getattr_getattr_L__mod___stages___2_____2___conv3_1x1_bn_act(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:886, code: x = self.drop_path(x)
    x_225 = self.getattr_getattr_L__mod___stages___2_____2___drop_path(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____2___shortcut = self.getattr_getattr_L__mod___stages___2_____2___shortcut(shortcut_7);  shortcut_7 = None
    x_226 = x_225 + getattr_getattr_l__mod___stages___2_____2___shortcut;  x_225 = getattr_getattr_l__mod___stages___2_____2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    shortcut_8 = self.getattr_getattr_L__mod___stages___2_____2___act(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_227 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_conv(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_227 = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_232 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:882, code: x = self.conv2_kxk(x)
    x_233 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    x_234 = self.getattr_getattr_L__mod___stages___3_____0___self_attn_qkv(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_2 = torch.functional.split(x_234, [512, 512, 512], dim = 1);  x_234 = None
    q_8 = split_2[0]
    k_4 = split_2[1]
    v_4 = split_2[2];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    reshape_24 = q_8.reshape(32, 128, -1);  q_8 = None
    q_9 = reshape_24.transpose(-1, -2);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    k_5 = k_4.reshape(32, 128, -1);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    reshape_26 = v_4.reshape(32, 128, -1);  v_4 = None
    v_5 = reshape_26.transpose(-1, -2);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    matmul_8 = q_9 @ k_5;  k_5 = None
    mul_8 = matmul_8 * 0.08838834764831845;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    q_10 = q_9.reshape(32, 16, 16, -1);  q_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:73, code: rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
    getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_width_rel = self.getattr_getattr_L__mod___stages___3_____0___self_attn_pos_embed_width_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_14 = getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_width_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_width_rel = None
    x_235 = q_10 @ transpose_14;  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_236 = x_235.reshape(-1, 16, 31);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_8 = torch.nn.functional.pad(x_236, [0, 1]);  x_236 = None
    x_pad_12 = pad_8.flatten(1);  pad_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_13 = torch.nn.functional.pad(x_pad_12, [0, 15]);  x_pad_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_14 = x_pad_13.reshape(-1, 17, 31);  x_pad_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_237 = x_pad_14[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))];  x_pad_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_30 = x_237.reshape(32, 16, 1, 16, 16);  x_237 = None
    x_238 = reshape_30.expand(-1, -1, 16, -1, -1);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_w_2 = x_238.permute((0, 1, 3, 2, 4));  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    q_11 = q_10.transpose(1, 2);  q_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:77, code: rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
    getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_height_rel = self.getattr_getattr_L__mod___stages___3_____0___self_attn_pos_embed_height_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_16 = getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_height_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_height_rel = None
    x_239 = q_11 @ transpose_16;  q_11 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_240 = x_239.reshape(-1, 16, 31);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_10 = torch.nn.functional.pad(x_240, [0, 1]);  x_240 = None
    x_pad_15 = pad_10.flatten(1);  pad_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_16 = torch.nn.functional.pad(x_pad_15, [0, 15]);  x_pad_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_17 = x_pad_16.reshape(-1, 17, 31);  x_pad_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_241 = x_pad_17[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))];  x_pad_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_33 = x_241.reshape(32, 16, 1, 16, 16);  x_241 = None
    x_242 = reshape_33.expand(-1, -1, 16, -1, -1);  reshape_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_h_2 = x_242.permute((0, 3, 1, 4, 2));  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    rel_logits_4 = rel_logits_h_2 + rel_logits_w_2;  rel_logits_h_2 = rel_logits_w_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    rel_logits_5 = rel_logits_4.reshape(32, 256, 256);  rel_logits_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    attn_4 = mul_8 + rel_logits_5;  mul_8 = rel_logits_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    attn_5 = attn_4.softmax(dim = -1);  attn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    matmul_11 = attn_5 @ v_5;  attn_5 = v_5 = None
    transpose_17 = matmul_11.transpose(-1, -2);  matmul_11 = None
    out_4 = transpose_17.reshape(8, 512, 16, 16);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    x_243 = self.getattr_getattr_L__mod___stages___3_____0___self_attn_pool(out_4);  out_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___post_attn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___post_attn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___post_attn_running_var = self.getattr_getattr_L__mod___stages___3_____0___post_attn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___post_attn_weight = self.getattr_getattr_L__mod___stages___3_____0___post_attn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___post_attn_bias = self.getattr_getattr_L__mod___stages___3_____0___post_attn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_244 = torch.nn.functional.batch_norm(x_243, getattr_getattr_l__mod___stages___3_____0___post_attn_running_mean, getattr_getattr_l__mod___stages___3_____0___post_attn_running_var, getattr_getattr_l__mod___stages___3_____0___post_attn_weight, getattr_getattr_l__mod___stages___3_____0___post_attn_bias, False, 0.1, 1e-05);  x_243 = getattr_getattr_l__mod___stages___3_____0___post_attn_running_mean = getattr_getattr_l__mod___stages___3_____0___post_attn_running_var = getattr_getattr_l__mod___stages___3_____0___post_attn_weight = getattr_getattr_l__mod___stages___3_____0___post_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_245 = self.getattr_getattr_L__mod___stages___3_____0___post_attn_drop(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_247 = self.getattr_getattr_L__mod___stages___3_____0___post_attn_act(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_248 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_conv(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_249 = torch.nn.functional.batch_norm(x_248, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_248 = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_250 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_drop(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_253 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_act(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:886, code: x = self.drop_path(x)
    x_254 = self.getattr_getattr_L__mod___stages___3_____0___drop_path(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_255 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_conv(shortcut_8);  shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_256 = torch.nn.functional.batch_norm(x_255, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias, False, 0.1, 1e-05);  x_255 = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_257 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_drop(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_259 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_act(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    x_260 = x_254 + x_259;  x_254 = x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    shortcut_9 = self.getattr_getattr_L__mod___stages___3_____0___act(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_261 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_conv(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_262 = torch.nn.functional.batch_norm(x_261, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias, False, 0.1, 1e-05);  x_261 = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_263 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_drop(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_266 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_act(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:882, code: x = self.conv2_kxk(x)
    x_267 = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    x_268 = self.getattr_getattr_L__mod___stages___3_____1___self_attn_qkv(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_3 = torch.functional.split(x_268, [512, 512, 512], dim = 1);  x_268 = None
    q_12 = split_3[0]
    k_6 = split_3[1]
    v_6 = split_3[2];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    reshape_36 = q_12.reshape(32, 128, -1);  q_12 = None
    q_13 = reshape_36.transpose(-1, -2);  reshape_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    k_7 = k_6.reshape(32, 128, -1);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    reshape_38 = v_6.reshape(32, 128, -1);  v_6 = None
    v_7 = reshape_38.transpose(-1, -2);  reshape_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    matmul_12 = q_13 @ k_7;  k_7 = None
    mul_9 = matmul_12 * 0.08838834764831845;  matmul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    q_14 = q_13.reshape(32, 8, 8, -1);  q_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:73, code: rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
    getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_width_rel = self.getattr_getattr_L__mod___stages___3_____1___self_attn_pos_embed_width_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_20 = getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_width_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_width_rel = None
    x_269 = q_14 @ transpose_20;  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_270 = x_269.reshape(-1, 8, 15);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_12 = torch.nn.functional.pad(x_270, [0, 1]);  x_270 = None
    x_pad_18 = pad_12.flatten(1);  pad_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_19 = torch.nn.functional.pad(x_pad_18, [0, 7]);  x_pad_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_20 = x_pad_19.reshape(-1, 9, 15);  x_pad_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_271 = x_pad_20[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))];  x_pad_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_42 = x_271.reshape(32, 8, 1, 8, 8);  x_271 = None
    x_272 = reshape_42.expand(-1, -1, 8, -1, -1);  reshape_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_w_3 = x_272.permute((0, 1, 3, 2, 4));  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    q_15 = q_14.transpose(1, 2);  q_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:77, code: rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
    getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_height_rel = self.getattr_getattr_L__mod___stages___3_____1___self_attn_pos_embed_height_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_22 = getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_height_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_height_rel = None
    x_273 = q_15 @ transpose_22;  q_15 = transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    x_274 = x_273.reshape(-1, 8, 15);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_14 = torch.nn.functional.pad(x_274, [0, 1]);  x_274 = None
    x_pad_21 = pad_14.flatten(1);  pad_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    x_pad_22 = torch.nn.functional.pad(x_pad_21, [0, 7]);  x_pad_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x_pad_23 = x_pad_22.reshape(-1, 9, 15);  x_pad_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    x_275 = x_pad_23[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))];  x_pad_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    reshape_45 = x_275.reshape(32, 8, 1, 8, 8);  x_275 = None
    x_276 = reshape_45.expand(-1, -1, 8, -1, -1);  reshape_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    rel_logits_h_3 = x_276.permute((0, 3, 1, 4, 2));  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    rel_logits_6 = rel_logits_h_3 + rel_logits_w_3;  rel_logits_h_3 = rel_logits_w_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    rel_logits_7 = rel_logits_6.reshape(32, 64, 64);  rel_logits_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    attn_6 = mul_9 + rel_logits_7;  mul_9 = rel_logits_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    attn_7 = attn_6.softmax(dim = -1);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    matmul_15 = attn_7 @ v_7;  attn_7 = v_7 = None
    transpose_23 = matmul_15.transpose(-1, -2);  matmul_15 = None
    out_6 = transpose_23.reshape(8, 512, 8, 8);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    x_277 = self.getattr_getattr_L__mod___stages___3_____1___self_attn_pool(out_6);  out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____1___post_attn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___post_attn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___post_attn_running_var = self.getattr_getattr_L__mod___stages___3_____1___post_attn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___post_attn_weight = self.getattr_getattr_L__mod___stages___3_____1___post_attn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___post_attn_bias = self.getattr_getattr_L__mod___stages___3_____1___post_attn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_278 = torch.nn.functional.batch_norm(x_277, getattr_getattr_l__mod___stages___3_____1___post_attn_running_mean, getattr_getattr_l__mod___stages___3_____1___post_attn_running_var, getattr_getattr_l__mod___stages___3_____1___post_attn_weight, getattr_getattr_l__mod___stages___3_____1___post_attn_bias, False, 0.1, 1e-05);  x_277 = getattr_getattr_l__mod___stages___3_____1___post_attn_running_mean = getattr_getattr_l__mod___stages___3_____1___post_attn_running_var = getattr_getattr_l__mod___stages___3_____1___post_attn_weight = getattr_getattr_l__mod___stages___3_____1___post_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_279 = self.getattr_getattr_L__mod___stages___3_____1___post_attn_drop(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_281 = self.getattr_getattr_L__mod___stages___3_____1___post_attn_act(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_282 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_conv(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_283 = torch.nn.functional.batch_norm(x_282, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias, False, 0.1, 1e-05);  x_282 = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_284 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_drop(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_287 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_act(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:886, code: x = self.drop_path(x)
    x_288 = self.getattr_getattr_L__mod___stages___3_____1___drop_path(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3_____1___shortcut = self.getattr_getattr_L__mod___stages___3_____1___shortcut(shortcut_9);  shortcut_9 = None
    x_289 = x_288 + getattr_getattr_l__mod___stages___3_____1___shortcut;  x_288 = getattr_getattr_l__mod___stages___3_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    x_290 = self.getattr_getattr_L__mod___stages___3_____1___act(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_291 = self.L__mod___final_conv_conv(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___final_conv_bn_running_mean = self.L__mod___final_conv_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___final_conv_bn_running_var = self.L__mod___final_conv_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___final_conv_bn_weight = self.L__mod___final_conv_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___final_conv_bn_bias = self.L__mod___final_conv_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_292 = torch.nn.functional.batch_norm(x_291, l__mod___final_conv_bn_running_mean, l__mod___final_conv_bn_running_var, l__mod___final_conv_bn_weight, l__mod___final_conv_bn_bias, False, 0.1, 1e-05);  x_291 = l__mod___final_conv_bn_running_mean = l__mod___final_conv_bn_running_var = l__mod___final_conv_bn_weight = l__mod___final_conv_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_293 = self.L__mod___final_conv_bn_drop(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_297 = self.L__mod___final_conv_bn_act(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_298 = self.L__mod___head_global_pool_pool(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_300 = self.L__mod___head_global_pool_flatten(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_301 = self.L__mod___head_drop(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_302 = self.L__mod___head_fc(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    x_303 = self.L__mod___head_flatten(x_302);  x_302 = None
    return (x_303,)
    