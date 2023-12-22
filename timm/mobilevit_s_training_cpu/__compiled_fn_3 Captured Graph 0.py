from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___stem_conv(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___stem_bn_num_batches_tracked = self.L__mod___stem_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_ = l__mod___stem_bn_num_batches_tracked.add_(1);  l__mod___stem_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___stem_bn_running_mean = self.L__mod___stem_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_bn_running_var = self.L__mod___stem_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_bn_weight = self.L__mod___stem_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_bn_bias = self.L__mod___stem_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___stem_bn_running_mean, l__mod___stem_bn_running_var, l__mod___stem_bn_weight, l__mod___stem_bn_bias, True, 0.1, 1e-05);  x = l__mod___stem_bn_running_mean = l__mod___stem_bn_running_var = l__mod___stem_bn_weight = l__mod___stem_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___stem_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___stem_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_6 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_conv(shortcut);  shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__1 = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_6 = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_11 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_12 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_conv(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__2 = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_13 = torch.nn.functional.batch_norm(x_12, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_12 = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_14 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_drop(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_17 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_18 = self.getattr_getattr_L__mod___stages___0_____0___conv2b_kxk(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_19 = self.getattr_getattr_L__mod___stages___0_____0___attn(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_20 = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_conv(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__3 = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_21 = torch.nn.functional.batch_norm(x_20, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_20 = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_22 = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_drop(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_25 = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_act(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_26 = self.getattr_getattr_L__mod___stages___0_____0___attn_last(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_27 = self.getattr_getattr_L__mod___stages___0_____0___drop_path(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_1 = self.getattr_getattr_L__mod___stages___0_____0___act(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_28 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_conv(shortcut_1);  shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__4 = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_29 = torch.nn.functional.batch_norm(x_28, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_28 = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_30 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_33 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_act(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_34 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_conv(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__5 = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_35 = torch.nn.functional.batch_norm(x_34, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_34 = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_36 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_drop(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_39 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_act(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_40 = self.getattr_getattr_L__mod___stages___1_____0___conv2b_kxk(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_41 = self.getattr_getattr_L__mod___stages___1_____0___attn(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_42 = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_conv(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__6 = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_43 = torch.nn.functional.batch_norm(x_42, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_42 = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_44 = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_drop(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_47 = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_48 = self.getattr_getattr_L__mod___stages___1_____0___attn_last(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_49 = self.getattr_getattr_L__mod___stages___1_____0___drop_path(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___stages___1_____0___act(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_50 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_conv(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__7 = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_51 = torch.nn.functional.batch_norm(x_50, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_50 = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_52 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_55 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_act(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_56 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_conv(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__8 = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_57 = torch.nn.functional.batch_norm(x_56, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_56 = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_58 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_drop(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_61 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_act(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_62 = self.getattr_getattr_L__mod___stages___1_____1___conv2b_kxk(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_63 = self.getattr_getattr_L__mod___stages___1_____1___attn(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_64 = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_conv(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__9 = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_65 = torch.nn.functional.batch_norm(x_64, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_64 = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_66 = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_drop(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_69 = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_act(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_70 = self.getattr_getattr_L__mod___stages___1_____1___attn_last(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_71 = self.getattr_getattr_L__mod___stages___1_____1___drop_path(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1_____1___shortcut = self.getattr_getattr_L__mod___stages___1_____1___shortcut(shortcut_2);  shortcut_2 = None
    x_72 = x_71 + getattr_getattr_l__mod___stages___1_____1___shortcut;  x_71 = getattr_getattr_l__mod___stages___1_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_3 = self.getattr_getattr_L__mod___stages___1_____1___act(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_73 = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_conv(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__10 = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_74 = torch.nn.functional.batch_norm(x_73, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_73 = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____2___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_75 = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_drop(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_78 = self.getattr_getattr_L__mod___stages___1_____2___conv1_1x1_bn_act(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_79 = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_conv(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__11 = getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_80 = torch.nn.functional.batch_norm(x_79, getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_79 = getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____2___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_81 = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_bn_drop(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_84 = self.getattr_getattr_L__mod___stages___1_____2___conv2_kxk_bn_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_85 = self.getattr_getattr_L__mod___stages___1_____2___conv2b_kxk(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_86 = self.getattr_getattr_L__mod___stages___1_____2___attn(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_87 = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_conv(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__12 = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_88 = torch.nn.functional.batch_norm(x_87, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_87 = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____2___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_89 = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_drop(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_92 = self.getattr_getattr_L__mod___stages___1_____2___conv3_1x1_bn_act(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_93 = self.getattr_getattr_L__mod___stages___1_____2___attn_last(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_94 = self.getattr_getattr_L__mod___stages___1_____2___drop_path(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___1_____2___shortcut = self.getattr_getattr_L__mod___stages___1_____2___shortcut(shortcut_3);  shortcut_3 = None
    x_95 = x_94 + getattr_getattr_l__mod___stages___1_____2___shortcut;  x_94 = getattr_getattr_l__mod___stages___1_____2___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___stages___1_____2___act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_96 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_conv(shortcut_4);  shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__13 = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_97 = torch.nn.functional.batch_norm(x_96, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_96 = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_98 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_drop(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_101 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_102 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_conv(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__14 = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_103 = torch.nn.functional.batch_norm(x_102, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_102 = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_104 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_drop(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_107 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_act(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_108 = self.getattr_getattr_L__mod___stages___2_____0___conv2b_kxk(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_109 = self.getattr_getattr_L__mod___stages___2_____0___attn(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_110 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_conv(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__15 = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_111 = torch.nn.functional.batch_norm(x_110, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_110 = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_112 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_drop(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_115 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_act(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_116 = self.getattr_getattr_L__mod___stages___2_____0___attn_last(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_117 = self.getattr_getattr_L__mod___stages___2_____0___drop_path(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_5 = self.getattr_getattr_L__mod___stages___2_____0___act(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_118 = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_conv(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__16 = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_119 = torch.nn.functional.batch_norm(x_118, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_bias, True, 0.1, 1e-05);  x_118 = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_120 = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_123 = self.getattr_getattr_L__mod___stages___2_____1___conv_kxk_bn_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    x_124 = self.getattr_getattr_L__mod___stages___2_____1___conv_1x1(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    reshape = x_124.reshape(18432, 2, 16, 2);  x_124 = None
    x_125 = reshape.transpose(1, 2);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    reshape_1 = x_125.reshape(8, 144, 256, 4);  x_125 = None
    transpose_1 = reshape_1.transpose(1, 3);  reshape_1 = None
    x_126 = transpose_1.reshape(32, 256, -1);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___norm1(x_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___attn_qkv(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1 = None
    reshape_3 = getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv.reshape(32, 256, 3, 4, 36);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv = None
    qkv = reshape_3.permute(2, 0, 3, 1, 4);  reshape_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___attn_q_norm(q);  q = None
    k_1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___attn_k_norm(k);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_127 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v, dropout_p = 0.0);  q_1 = k_1 = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_2 = x_127.transpose(1, 2);  x_127 = None
    x_128 = transpose_2.reshape(32, 256, 144);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_129 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___attn_proj(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_130 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___attn_proj_drop(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___ls1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___ls1(x_130);  x_130 = None
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___drop_path1(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___ls1);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___ls1 = None
    x_131 = x_126 + getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___drop_path1;  x_126 = getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___norm2(x_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_132 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___mlp_fc1(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_133 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___mlp_act(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_134 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___mlp_drop1(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_135 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___mlp_norm(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_136 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___mlp_fc2(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_137 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___mlp_drop2(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___ls2 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___ls2(x_137);  x_137 = None
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___0___drop_path2(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___ls2);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___ls2 = None
    x_138 = x_131 + getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___drop_path2;  x_131 = getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___norm1(x_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___attn_qkv(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1 = None
    reshape_5 = getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv.reshape(32, 256, 3, 4, 36);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv = None
    qkv_1 = reshape_5.permute(2, 0, 3, 1, 4);  reshape_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_2 = unbind_1[0]
    k_2 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_3 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___attn_q_norm(q_2);  q_2 = None
    k_3 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___attn_k_norm(k_2);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_139 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_1, dropout_p = 0.0);  q_3 = k_3 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_3 = x_139.transpose(1, 2);  x_139 = None
    x_140 = transpose_3.reshape(32, 256, 144);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_141 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___attn_proj(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_142 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___attn_proj_drop(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___ls1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___ls1(x_142);  x_142 = None
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___drop_path1(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___ls1);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___ls1 = None
    x_143 = x_138 + getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___drop_path1;  x_138 = getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___norm2(x_143)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_144 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___mlp_fc1(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_145 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___mlp_act(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_146 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___mlp_drop1(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_147 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___mlp_norm(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_148 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___mlp_fc2(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_149 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___mlp_drop2(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___ls2 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___ls2(x_149);  x_149 = None
    getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___2_____1___transformer___1___drop_path2(getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___ls2);  getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___ls2 = None
    x_151 = x_143 + getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___drop_path2;  x_143 = getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    x_152 = self.getattr_getattr_L__mod___stages___2_____1___norm(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    contiguous = x_152.contiguous();  x_152 = None
    x_153 = contiguous.view(8, 4, 256, -1);  contiguous = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    transpose_4 = x_153.transpose(1, 3);  x_153 = None
    x_154 = transpose_4.reshape(18432, 16, 2, 2);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    transpose_5 = x_154.transpose(1, 2);  x_154 = None
    x_155 = transpose_5.reshape(8, 144, 32, 32);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_156 = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_conv(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__17 = getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_157 = torch.nn.functional.batch_norm(x_156, getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_bias, True, 0.1, 1e-05);  x_156 = getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv_proj_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_158 = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_bn_drop(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_161 = self.getattr_getattr_L__mod___stages___2_____1___conv_proj_bn_act(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat = torch.cat((shortcut_5, x_161), dim = 1);  shortcut_5 = x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_162 = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_conv(cat);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__18 = getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_163 = torch.nn.functional.batch_norm(x_162, getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_bias, True, 0.1, 1e-05);  x_162 = getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv_fusion_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_164 = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_bn_drop(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_6 = self.getattr_getattr_L__mod___stages___2_____1___conv_fusion_bn_act(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_168 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_conv(shortcut_6);  shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__19 = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_169 = torch.nn.functional.batch_norm(x_168, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_168 = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_170 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_drop(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_173 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_act(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_174 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_conv(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__20 = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_175 = torch.nn.functional.batch_norm(x_174, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_174 = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_176 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_drop(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_179 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk_bn_act(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_180 = self.getattr_getattr_L__mod___stages___3_____0___conv2b_kxk(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_181 = self.getattr_getattr_L__mod___stages___3_____0___attn(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_182 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_conv(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__21 = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_183 = torch.nn.functional.batch_norm(x_182, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_182 = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_184 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_drop(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_187 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_act(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_188 = self.getattr_getattr_L__mod___stages___3_____0___attn_last(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_189 = self.getattr_getattr_L__mod___stages___3_____0___drop_path(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_7 = self.getattr_getattr_L__mod___stages___3_____0___act(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_190 = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__22 = getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_191 = torch.nn.functional.batch_norm(x_190, getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_bias, True, 0.1, 1e-05);  x_190 = getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_192 = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_bn_drop(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_195 = self.getattr_getattr_L__mod___stages___3_____1___conv_kxk_bn_act(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    x_196 = self.getattr_getattr_L__mod___stages___3_____1___conv_1x1(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    reshape_9 = x_196.reshape(12288, 2, 8, 2);  x_196 = None
    x_197 = reshape_9.transpose(1, 2);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    reshape_10 = x_197.reshape(8, 192, 64, 4);  x_197 = None
    transpose_7 = reshape_10.transpose(1, 3);  reshape_10 = None
    x_198 = transpose_7.reshape(32, 64, -1);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___norm1(x_198)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___attn_qkv(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 = None
    reshape_12 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv.reshape(32, 64, 3, 4, 48);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv = None
    qkv_2 = reshape_12.permute(2, 0, 3, 1, 4);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_4 = unbind_2[0]
    k_4 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_5 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___attn_q_norm(q_4);  q_4 = None
    k_5 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___attn_k_norm(k_4);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_199 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_2, dropout_p = 0.0);  q_5 = k_5 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_8 = x_199.transpose(1, 2);  x_199 = None
    x_200 = transpose_8.reshape(32, 64, 192);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_201 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___attn_proj(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_202 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___attn_proj_drop(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___ls1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___ls1(x_202);  x_202 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___drop_path1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___ls1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___ls1 = None
    x_203 = x_198 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___drop_path1;  x_198 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___norm2(x_203)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_204 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___mlp_fc1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_205 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___mlp_act(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_206 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___mlp_drop1(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_207 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___mlp_norm(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_208 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___mlp_fc2(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_209 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___mlp_drop2(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___ls2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___ls2(x_209);  x_209 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___0___drop_path2(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___ls2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___ls2 = None
    x_210 = x_203 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___drop_path2;  x_203 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___norm1(x_210)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___attn_qkv(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1 = None
    reshape_14 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv.reshape(32, 64, 3, 4, 48);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv = None
    qkv_3 = reshape_14.permute(2, 0, 3, 1, 4);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_6 = unbind_3[0]
    k_6 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_7 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___attn_q_norm(q_6);  q_6 = None
    k_7 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___attn_k_norm(k_6);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_211 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_3, dropout_p = 0.0);  q_7 = k_7 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_9 = x_211.transpose(1, 2);  x_211 = None
    x_212 = transpose_9.reshape(32, 64, 192);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_213 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___attn_proj(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_214 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___attn_proj_drop(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___ls1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___ls1(x_214);  x_214 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___drop_path1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___ls1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___ls1 = None
    x_215 = x_210 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___drop_path1;  x_210 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___norm2(x_215)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_216 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___mlp_fc1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_217 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___mlp_act(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_218 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___mlp_drop1(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_219 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___mlp_norm(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_220 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___mlp_fc2(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_221 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___mlp_drop2(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___ls2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___ls2(x_221);  x_221 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___1___drop_path2(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___ls2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___ls2 = None
    x_222 = x_215 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___drop_path2;  x_215 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___norm1(x_222)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___attn_qkv(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1 = None
    reshape_16 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv.reshape(32, 64, 3, 4, 48);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv = None
    qkv_4 = reshape_16.permute(2, 0, 3, 1, 4);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_8 = unbind_4[0]
    k_8 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_9 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___attn_q_norm(q_8);  q_8 = None
    k_9 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___attn_k_norm(k_8);  k_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_223 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_4, dropout_p = 0.0);  q_9 = k_9 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_10 = x_223.transpose(1, 2);  x_223 = None
    x_224 = transpose_10.reshape(32, 64, 192);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_225 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___attn_proj(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_226 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___attn_proj_drop(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___ls1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___ls1(x_226);  x_226 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___drop_path1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___ls1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___ls1 = None
    x_227 = x_222 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___drop_path1;  x_222 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___norm2(x_227)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_228 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___mlp_fc1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_229 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___mlp_act(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_230 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___mlp_drop1(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_231 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___mlp_norm(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_232 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___mlp_fc2(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_233 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___mlp_drop2(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___ls2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___ls2(x_233);  x_233 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___2___drop_path2(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___ls2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___ls2 = None
    x_234 = x_227 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___drop_path2;  x_227 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___norm1(x_234)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___attn_qkv(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm1 = None
    reshape_18 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv.reshape(32, 64, 3, 4, 48);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv = None
    qkv_5 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_10 = unbind_5[0]
    k_10 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_11 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___attn_q_norm(q_10);  q_10 = None
    k_11 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___attn_k_norm(k_10);  k_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_235 = torch._C._nn.scaled_dot_product_attention(q_11, k_11, v_5, dropout_p = 0.0);  q_11 = k_11 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_11 = x_235.transpose(1, 2);  x_235 = None
    x_236 = transpose_11.reshape(32, 64, 192);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_237 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___attn_proj(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_238 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___attn_proj_drop(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___ls1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___ls1(x_238);  x_238 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___drop_path1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___ls1);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___ls1 = None
    x_239 = x_234 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___drop_path1;  x_234 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___norm2(x_239)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_240 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___mlp_fc1(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_241 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___mlp_act(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_242 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___mlp_drop1(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_243 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___mlp_norm(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_244 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___mlp_fc2(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_245 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___mlp_drop2(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___ls2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___ls2(x_245);  x_245 = None
    getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___3_____1___transformer___3___drop_path2(getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___ls2);  getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___ls2 = None
    x_247 = x_239 + getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___drop_path2;  x_239 = getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    x_248 = self.getattr_getattr_L__mod___stages___3_____1___norm(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    contiguous_1 = x_248.contiguous();  x_248 = None
    x_249 = contiguous_1.view(8, 4, 64, -1);  contiguous_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    transpose_12 = x_249.transpose(1, 3);  x_249 = None
    x_250 = transpose_12.reshape(12288, 8, 2, 2);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    transpose_13 = x_250.transpose(1, 2);  x_250 = None
    x_251 = transpose_13.reshape(8, 192, 16, 16);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_252 = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_conv(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__23 = getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_253 = torch.nn.functional.batch_norm(x_252, getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_bias, True, 0.1, 1e-05);  x_252 = getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv_proj_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_254 = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_bn_drop(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_257 = self.getattr_getattr_L__mod___stages___3_____1___conv_proj_bn_act(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_1 = torch.cat((shortcut_7, x_257), dim = 1);  shortcut_7 = x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_258 = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_conv(cat_1);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__24 = getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_259 = torch.nn.functional.batch_norm(x_258, getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_bias, True, 0.1, 1e-05);  x_258 = getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv_fusion_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_260 = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_bn_drop(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_8 = self.getattr_getattr_L__mod___stages___3_____1___conv_fusion_bn_act(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_264 = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_conv(shortcut_8);  shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__25 = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_265 = torch.nn.functional.batch_norm(x_264, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_264 = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_266 = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_drop(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_269 = self.getattr_getattr_L__mod___stages___4_____0___conv1_1x1_bn_act(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_270 = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_conv(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__26 = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_271 = torch.nn.functional.batch_norm(x_270, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_270 = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___4_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_272 = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_drop(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_275 = self.getattr_getattr_L__mod___stages___4_____0___conv2_kxk_bn_act(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_276 = self.getattr_getattr_L__mod___stages___4_____0___conv2b_kxk(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:331, code: x = self.attn(x)
    x_277 = self.getattr_getattr_L__mod___stages___4_____0___attn(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_278 = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_conv(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__27 = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_279 = torch.nn.functional.batch_norm(x_278, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_278 = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___4_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_280 = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_drop(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_283 = self.getattr_getattr_L__mod___stages___4_____0___conv3_1x1_bn_act(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_284 = self.getattr_getattr_L__mod___stages___4_____0___attn_last(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_285 = self.getattr_getattr_L__mod___stages___4_____0___drop_path(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_9 = self.getattr_getattr_L__mod___stages___4_____0___act(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_286 = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_conv(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__28 = getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_weight = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_bias = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_287 = torch.nn.functional.batch_norm(x_286, getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_running_mean, getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_running_var, getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_weight, getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_bias, True, 0.1, 1e-05);  x_286 = getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_running_mean = getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_running_var = getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_weight = getattr_getattr_l__mod___stages___4_____1___conv_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_288 = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_bn_drop(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_291 = self.getattr_getattr_L__mod___stages___4_____1___conv_kxk_bn_act(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    x_292 = self.getattr_getattr_L__mod___stages___4_____1___conv_1x1(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    reshape_22 = x_292.reshape(7680, 2, 4, 2);  x_292 = None
    x_293 = reshape_22.transpose(1, 2);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    reshape_23 = x_293.reshape(8, 240, 16, 4);  x_293 = None
    transpose_15 = reshape_23.transpose(1, 3);  reshape_23 = None
    x_294 = transpose_15.reshape(32, 16, -1);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___norm1(x_294)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___attn_qkv(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 = None
    reshape_25 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv.reshape(32, 16, 3, 4, 60);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv = None
    qkv_6 = reshape_25.permute(2, 0, 3, 1, 4);  reshape_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_12 = unbind_6[0]
    k_12 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_13 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___attn_q_norm(q_12);  q_12 = None
    k_13 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___attn_k_norm(k_12);  k_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_295 = torch._C._nn.scaled_dot_product_attention(q_13, k_13, v_6, dropout_p = 0.0);  q_13 = k_13 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_16 = x_295.transpose(1, 2);  x_295 = None
    x_296 = transpose_16.reshape(32, 16, 240);  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_297 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___attn_proj(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_298 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___attn_proj_drop(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___ls1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___ls1(x_298);  x_298 = None
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___drop_path1(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___ls1);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___ls1 = None
    x_299 = x_294 + getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___drop_path1;  x_294 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___norm2(x_299)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_300 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___mlp_fc1(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_301 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___mlp_act(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_302 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___mlp_drop1(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_303 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___mlp_norm(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_304 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___mlp_fc2(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_305 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___mlp_drop2(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___ls2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___ls2(x_305);  x_305 = None
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___0___drop_path2(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___ls2);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___ls2 = None
    x_306 = x_299 + getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___drop_path2;  x_299 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___norm1(x_306)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___attn_qkv(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1 = None
    reshape_27 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv.reshape(32, 16, 3, 4, 60);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv = None
    qkv_7 = reshape_27.permute(2, 0, 3, 1, 4);  reshape_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_14 = unbind_7[0]
    k_14 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_15 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___attn_q_norm(q_14);  q_14 = None
    k_15 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___attn_k_norm(k_14);  k_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_307 = torch._C._nn.scaled_dot_product_attention(q_15, k_15, v_7, dropout_p = 0.0);  q_15 = k_15 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_17 = x_307.transpose(1, 2);  x_307 = None
    x_308 = transpose_17.reshape(32, 16, 240);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_309 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___attn_proj(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_310 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___attn_proj_drop(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___ls1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___ls1(x_310);  x_310 = None
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___drop_path1(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___ls1);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___ls1 = None
    x_311 = x_306 + getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___drop_path1;  x_306 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___norm2(x_311)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_312 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___mlp_fc1(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_313 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___mlp_act(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_314 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___mlp_drop1(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_315 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___mlp_norm(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_316 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___mlp_fc2(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_317 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___mlp_drop2(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___ls2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___ls2(x_317);  x_317 = None
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___1___drop_path2(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___ls2);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___ls2 = None
    x_318 = x_311 + getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___drop_path2;  x_311 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___norm1(x_318)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___attn_qkv(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1 = None
    reshape_29 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv.reshape(32, 16, 3, 4, 60);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv = None
    qkv_8 = reshape_29.permute(2, 0, 3, 1, 4);  reshape_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_16 = unbind_8[0]
    k_16 = unbind_8[1]
    v_8 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_17 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___attn_q_norm(q_16);  q_16 = None
    k_17 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___attn_k_norm(k_16);  k_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_319 = torch._C._nn.scaled_dot_product_attention(q_17, k_17, v_8, dropout_p = 0.0);  q_17 = k_17 = v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_18 = x_319.transpose(1, 2);  x_319 = None
    x_320 = transpose_18.reshape(32, 16, 240);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_321 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___attn_proj(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_322 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___attn_proj_drop(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___ls1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___ls1(x_322);  x_322 = None
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___drop_path1 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___drop_path1(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___ls1);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___ls1 = None
    x_323 = x_318 + getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___drop_path1;  x_318 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___norm2(x_323)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_324 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___mlp_fc1(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm2);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_325 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___mlp_act(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_326 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___mlp_drop1(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_327 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___mlp_norm(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_328 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___mlp_fc2(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_329 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___mlp_drop2(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___ls2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___ls2(x_329);  x_329 = None
    getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___drop_path2 = self.getattr_getattr_getattr_L__mod___stages___4_____1___transformer___2___drop_path2(getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___ls2);  getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___ls2 = None
    x_331 = x_323 + getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___drop_path2;  x_323 = getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    x_332 = self.getattr_getattr_L__mod___stages___4_____1___norm(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    contiguous_2 = x_332.contiguous();  x_332 = None
    x_333 = contiguous_2.view(8, 4, 16, -1);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    transpose_19 = x_333.transpose(1, 3);  x_333 = None
    x_334 = transpose_19.reshape(7680, 4, 2, 2);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    transpose_20 = x_334.transpose(1, 2);  x_334 = None
    x_335 = transpose_20.reshape(8, 240, 8, 8);  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_336 = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_conv(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__29 = getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_running_var = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_weight = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_bias = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_337 = torch.nn.functional.batch_norm(x_336, getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_running_mean, getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_running_var, getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_weight, getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_bias, True, 0.1, 1e-05);  x_336 = getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_running_mean = getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_running_var = getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_weight = getattr_getattr_l__mod___stages___4_____1___conv_proj_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_338 = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_bn_drop(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_341 = self.getattr_getattr_L__mod___stages___4_____1___conv_proj_bn_act(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_2 = torch.cat((shortcut_9, x_341), dim = 1);  shortcut_9 = x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_342 = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_conv(cat_2);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__30 = getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_running_mean = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_running_var = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_weight = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_bias = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_343 = torch.nn.functional.batch_norm(x_342, getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_running_mean, getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_running_var, getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_weight, getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_bias, True, 0.1, 1e-05);  x_342 = getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_running_mean = getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_running_var = getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_weight = getattr_getattr_l__mod___stages___4_____1___conv_fusion_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_344 = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_bn_drop(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_348 = self.getattr_getattr_L__mod___stages___4_____1___conv_fusion_bn_act(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_349 = self.L__mod___final_conv_conv(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___final_conv_bn_num_batches_tracked = self.L__mod___final_conv_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__31 = l__mod___final_conv_bn_num_batches_tracked.add_(1);  l__mod___final_conv_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___final_conv_bn_running_mean = self.L__mod___final_conv_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___final_conv_bn_running_var = self.L__mod___final_conv_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___final_conv_bn_weight = self.L__mod___final_conv_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___final_conv_bn_bias = self.L__mod___final_conv_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_350 = torch.nn.functional.batch_norm(x_349, l__mod___final_conv_bn_running_mean, l__mod___final_conv_bn_running_var, l__mod___final_conv_bn_weight, l__mod___final_conv_bn_bias, True, 0.1, 1e-05);  x_349 = l__mod___final_conv_bn_running_mean = l__mod___final_conv_bn_running_var = l__mod___final_conv_bn_weight = l__mod___final_conv_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_351 = self.L__mod___final_conv_bn_drop(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_355 = self.L__mod___final_conv_bn_act(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_356 = self.L__mod___head_global_pool_pool(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_358 = self.L__mod___head_global_pool_flatten(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_359 = self.L__mod___head_drop(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_360 = self.L__mod___head_fc(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    pred = self.L__mod___head_flatten(x_360);  x_360 = None
    return (pred,)
    