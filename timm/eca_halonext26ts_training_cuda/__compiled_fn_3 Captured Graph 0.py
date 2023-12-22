from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___stem_conv1_conv(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___stem_conv1_bn_num_batches_tracked = self.L__mod___stem_conv1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_ = l__mod___stem_conv1_bn_num_batches_tracked.add_(1);  l__mod___stem_conv1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___stem_conv1_bn_running_mean = self.L__mod___stem_conv1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv1_bn_running_var = self.L__mod___stem_conv1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv1_bn_weight = self.L__mod___stem_conv1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv1_bn_bias = self.L__mod___stem_conv1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___stem_conv1_bn_running_mean, l__mod___stem_conv1_bn_running_var, l__mod___stem_conv1_bn_weight, l__mod___stem_conv1_bn_bias, True, 0.1, 1e-05);  x = l__mod___stem_conv1_bn_running_mean = l__mod___stem_conv1_bn_running_var = l__mod___stem_conv1_bn_weight = l__mod___stem_conv1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___stem_conv1_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_4 = self.L__mod___stem_conv1_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_5 = self.L__mod___stem_conv2_conv(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___stem_conv2_bn_num_batches_tracked = self.L__mod___stem_conv2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__1 = l__mod___stem_conv2_bn_num_batches_tracked.add_(1);  l__mod___stem_conv2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___stem_conv2_bn_running_mean = self.L__mod___stem_conv2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv2_bn_running_var = self.L__mod___stem_conv2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv2_bn_weight = self.L__mod___stem_conv2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv2_bn_bias = self.L__mod___stem_conv2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_6 = torch.nn.functional.batch_norm(x_5, l__mod___stem_conv2_bn_running_mean, l__mod___stem_conv2_bn_running_var, l__mod___stem_conv2_bn_weight, l__mod___stem_conv2_bn_bias, True, 0.1, 1e-05);  x_5 = l__mod___stem_conv2_bn_running_mean = l__mod___stem_conv2_bn_running_var = l__mod___stem_conv2_bn_weight = l__mod___stem_conv2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_7 = self.L__mod___stem_conv2_bn_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_9 = self.L__mod___stem_conv2_bn_act(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_10 = self.L__mod___stem_conv3_conv(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___stem_conv3_bn_num_batches_tracked = self.L__mod___stem_conv3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__2 = l__mod___stem_conv3_bn_num_batches_tracked.add_(1);  l__mod___stem_conv3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___stem_conv3_bn_running_mean = self.L__mod___stem_conv3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___stem_conv3_bn_running_var = self.L__mod___stem_conv3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___stem_conv3_bn_weight = self.L__mod___stem_conv3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___stem_conv3_bn_bias = self.L__mod___stem_conv3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_11 = torch.nn.functional.batch_norm(x_10, l__mod___stem_conv3_bn_running_mean, l__mod___stem_conv3_bn_running_var, l__mod___stem_conv3_bn_weight, l__mod___stem_conv3_bn_bias, True, 0.1, 1e-05);  x_10 = l__mod___stem_conv3_bn_running_mean = l__mod___stem_conv3_bn_running_var = l__mod___stem_conv3_bn_weight = l__mod___stem_conv3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_12 = self.L__mod___stem_conv3_bn_drop(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_14 = self.L__mod___stem_conv3_bn_act(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    shortcut = self.L__mod___stem_pool(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_16 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_conv(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__3 = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_17 = torch.nn.functional.batch_norm(x_16, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_16 = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_18 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_21 = self.getattr_getattr_L__mod___stages___0_____0___conv1_1x1_bn_act(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_22 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_conv(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__4 = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_23 = torch.nn.functional.batch_norm(x_22, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_22 = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_24 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_drop(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_27 = self.getattr_getattr_L__mod___stages___0_____0___conv2_kxk_bn_act(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_28 = self.getattr_getattr_L__mod___stages___0_____0___conv2b_kxk(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean = x_28.mean((2, 3))
    y = mean.view(8, 1, -1);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    y_1 = self.getattr_getattr_L__mod___stages___0_____0___attn_conv(y);  y = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid = y_1.sigmoid();  y_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    y_2 = sigmoid.view(8, -1, 1, 1);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_as = y_2.expand_as(x_28);  y_2 = None
    x_29 = x_28 * expand_as;  x_28 = expand_as = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_30 = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_conv(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__5 = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_31 = torch.nn.functional.batch_norm(x_30, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_30 = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____0___conv3_1x1_bn_bias = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__6 = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____0___shortcut_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___0_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_39 = torch.nn.functional.batch_norm(x_38, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias, True, 0.1, 1e-05);  x_38 = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___0_____0___shortcut_bn_bias = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__7 = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_45 = torch.nn.functional.batch_norm(x_44, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_44 = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_46 = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_drop(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_49 = self.getattr_getattr_L__mod___stages___0_____1___conv1_1x1_bn_act(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_50 = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_conv(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__8 = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_51 = torch.nn.functional.batch_norm(x_50, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_50 = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_52 = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_55 = self.getattr_getattr_L__mod___stages___0_____1___conv2_kxk_bn_act(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_56 = self.getattr_getattr_L__mod___stages___0_____1___conv2b_kxk(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_1 = x_56.mean((2, 3))
    y_3 = mean_1.view(8, 1, -1);  mean_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    y_4 = self.getattr_getattr_L__mod___stages___0_____1___attn_conv(y_3);  y_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1 = y_4.sigmoid();  y_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    y_5 = sigmoid_1.view(8, -1, 1, 1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_as_1 = y_5.expand_as(x_56);  y_5 = None
    x_57 = x_56 * expand_as_1;  x_56 = expand_as_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_58 = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_conv(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__9 = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___0_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_59 = torch.nn.functional.batch_norm(x_58, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_58 = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___0_____1___conv3_1x1_bn_bias = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__10 = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_68 = torch.nn.functional.batch_norm(x_67, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_67 = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_69 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_drop(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_72 = self.getattr_getattr_L__mod___stages___1_____0___conv1_1x1_bn_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_73 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_conv(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__11 = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_74 = torch.nn.functional.batch_norm(x_73, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_73 = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_75 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_drop(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_78 = self.getattr_getattr_L__mod___stages___1_____0___conv2_kxk_bn_act(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_79 = self.getattr_getattr_L__mod___stages___1_____0___conv2b_kxk(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_2 = x_79.mean((2, 3))
    y_6 = mean_2.view(8, 1, -1);  mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    y_7 = self.getattr_getattr_L__mod___stages___1_____0___attn_conv(y_6);  y_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2 = y_7.sigmoid();  y_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    y_8 = sigmoid_2.view(8, -1, 1, 1);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_as_2 = y_8.expand_as(x_79);  y_8 = None
    x_80 = x_79 * expand_as_2;  x_79 = expand_as_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_81 = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_conv(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__12 = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_82 = torch.nn.functional.batch_norm(x_81, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_81 = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____0___conv3_1x1_bn_bias = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__13 = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____0___shortcut_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___1_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_90 = torch.nn.functional.batch_norm(x_89, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias, True, 0.1, 1e-05);  x_89 = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___1_____0___shortcut_bn_bias = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__14 = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_96 = torch.nn.functional.batch_norm(x_95, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_95 = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_97 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_drop(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_100 = self.getattr_getattr_L__mod___stages___1_____1___conv1_1x1_bn_act(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_101 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_conv(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__15 = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_102 = torch.nn.functional.batch_norm(x_101, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_101 = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_103 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_drop(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_106 = self.getattr_getattr_L__mod___stages___1_____1___conv2_kxk_bn_act(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_107 = self.getattr_getattr_L__mod___stages___1_____1___conv2b_kxk(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_3 = x_107.mean((2, 3))
    y_9 = mean_3.view(8, 1, -1);  mean_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    y_10 = self.getattr_getattr_L__mod___stages___1_____1___attn_conv(y_9);  y_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3 = y_10.sigmoid();  y_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    y_11 = sigmoid_3.view(8, -1, 1, 1);  sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_as_3 = y_11.expand_as(x_107);  y_11 = None
    x_108 = x_107 * expand_as_3;  x_107 = expand_as_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_109 = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_conv(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__16 = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___1_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_110 = torch.nn.functional.batch_norm(x_109, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_109 = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___1_____1___conv3_1x1_bn_bias = None
    
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
    x_118 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_conv(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__17 = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_119 = torch.nn.functional.batch_norm(x_118, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_118 = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_120 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_123 = self.getattr_getattr_L__mod___stages___2_____0___conv1_1x1_bn_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_124 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_conv(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__18 = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_125 = torch.nn.functional.batch_norm(x_124, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias, True, 0.1, 1e-05);  x_124 = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv2_kxk_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_126 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_drop(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_129 = self.getattr_getattr_L__mod___stages___2_____0___conv2_kxk_bn_act(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:330, code: x = self.conv2b_kxk(x)
    x_130 = self.getattr_getattr_L__mod___stages___2_____0___conv2b_kxk(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:85, code: y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
    mean_4 = x_130.mean((2, 3))
    y_12 = mean_4.view(8, 1, -1);  mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:86, code: y = self.conv(y)
    y_13 = self.getattr_getattr_L__mod___stages___2_____0___attn_conv(y_12);  y_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4 = y_13.sigmoid();  y_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:90, code: y = self.gate(y).view(x.shape[0], -1, 1, 1)
    y_14 = sigmoid_4.view(8, -1, 1, 1);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/eca.py:91, code: return x * y.expand_as(x)
    expand_as_4 = y_14.expand_as(x_130);  y_14 = None
    x_131 = x_130 * expand_as_4;  x_130 = expand_as_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_132 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_conv(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__19 = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_133 = torch.nn.functional.batch_norm(x_132, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_132 = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_134 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_drop(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_137 = self.getattr_getattr_L__mod___stages___2_____0___conv3_1x1_bn_act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:333, code: x = self.attn_last(x)
    x_138 = self.getattr_getattr_L__mod___stages___2_____0___attn_last(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:334, code: x = self.drop_path(x)
    x_139 = self.getattr_getattr_L__mod___stages___2_____0___drop_path(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_140 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_conv(shortcut_4);  shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__20 = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____0___shortcut_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_141 = torch.nn.functional.batch_norm(x_140, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias, True, 0.1, 1e-05);  x_140 = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___2_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_142 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_drop(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_144 = self.getattr_getattr_L__mod___stages___2_____0___shortcut_bn_act(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    x_145 = x_139 + x_144;  x_139 = x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    shortcut_5 = self.getattr_getattr_L__mod___stages___2_____0___act(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_146 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_conv(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__21 = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_147 = torch.nn.functional.batch_norm(x_146, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_146 = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_148 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_drop(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_151 = self.getattr_getattr_L__mod___stages___2_____1___conv1_1x1_bn_act(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:882, code: x = self.conv2_kxk(x)
    x_152 = self.getattr_getattr_L__mod___stages___2_____1___conv2_kxk(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    q = self.getattr_getattr_L__mod___stages___2_____1___self_attn_q(x_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    reshape = q.reshape(-1, 16, 2, 8, 2, 8);  q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    q_1 = reshape.permute(0, 1, 3, 5, 2, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    reshape_1 = q_1.reshape(64, 16, -1, 4);  q_1 = None
    q_2 = reshape_1.transpose(1, 3);  reshape_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    kv = self.getattr_getattr_L__mod___stages___2_____1___self_attn_kv(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    kv_1 = torch.nn.functional.pad(kv, [2, 2, 2, 2]);  kv = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    unfold = kv_1.unfold(2, 12, 8);  kv_1 = None
    unfold_1 = unfold.unfold(3, 12, 8);  unfold = None
    reshape_2 = unfold_1.reshape(64, 48, 4, -1);  unfold_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    kv_2 = reshape_2.permute(0, 2, 3, 1);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    split = torch.functional.split(kv_2, [16, 32], dim = -1);  kv_2 = None
    k = split[0]
    v = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    transpose_1 = k.transpose(-1, -2);  k = None
    matmul = q_2 @ transpose_1;  transpose_1 = None
    mul_5 = matmul * 0.25;  matmul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    q_3 = q_2.reshape(-1, 8, 8, 16);  q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:86, code: rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
    getattr_getattr_l__mod___stages___2_____1___self_attn_pos_embed_width_rel = self.getattr_getattr_L__mod___stages___2_____1___self_attn_pos_embed_width_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_2 = getattr_getattr_l__mod___stages___2_____1___self_attn_pos_embed_width_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___2_____1___self_attn_pos_embed_width_rel = None
    x_153 = q_3 @ transpose_2;  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    x_154 = x_153.reshape(-1, 8, 23);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_1 = torch.nn.functional.pad(x_154, [0, 1]);  x_154 = None
    x_pad = pad_1.flatten(1);  pad_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    x_pad_1 = torch.nn.functional.pad(x_pad, [0, 15]);  x_pad = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x_pad_2 = x_pad_1.reshape(-1, 9, 23);  x_pad_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    x_155 = x_pad_2[(slice(None, None, None), slice(None, 8, None), slice(11, None, None))];  x_pad_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    reshape_6 = x_155.reshape(256, 8, 1, 8, 12);  x_155 = None
    x_156 = reshape_6.expand(-1, -1, 12, -1, -1);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    rel_logits_w = x_156.permute((0, 1, 3, 2, 4));  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    q_4 = q_3.transpose(1, 2);  q_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:90, code: rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
    getattr_getattr_l__mod___stages___2_____1___self_attn_pos_embed_height_rel = self.getattr_getattr_L__mod___stages___2_____1___self_attn_pos_embed_height_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_4 = getattr_getattr_l__mod___stages___2_____1___self_attn_pos_embed_height_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___2_____1___self_attn_pos_embed_height_rel = None
    x_157 = q_4 @ transpose_4;  q_4 = transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    x_158 = x_157.reshape(-1, 8, 23);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_3 = torch.nn.functional.pad(x_158, [0, 1]);  x_158 = None
    x_pad_3 = pad_3.flatten(1);  pad_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    x_pad_4 = torch.nn.functional.pad(x_pad_3, [0, 15]);  x_pad_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x_pad_5 = x_pad_4.reshape(-1, 9, 23);  x_pad_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    x_159 = x_pad_5[(slice(None, None, None), slice(None, 8, None), slice(11, None, None))];  x_pad_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    reshape_9 = x_159.reshape(256, 8, 1, 8, 12);  x_159 = None
    x_160 = reshape_9.expand(-1, -1, 12, -1, -1);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    rel_logits_h = x_160.permute((0, 3, 1, 4, 2));  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:92, code: rel_logits = rel_logits_h + rel_logits_w
    rel_logits = rel_logits_h + rel_logits_w;  rel_logits_h = rel_logits_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    rel_logits_1 = rel_logits.reshape(64, 4, 64, -1);  rel_logits = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    attn = mul_5 + rel_logits_1;  mul_5 = rel_logits_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    attn_1 = attn.softmax(dim = -1);  attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    matmul_3 = attn_1 @ v;  attn_1 = v = None
    out = matmul_3.transpose(1, 3);  matmul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    out_1 = out.reshape(-1, 8, 8, 2, 2);  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    permute_4 = out_1.permute(0, 3, 1, 4, 2);  out_1 = None
    contiguous = permute_4.contiguous();  permute_4 = None
    out_2 = contiguous.view(8, 256, 16, 16);  contiguous = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:209, code: out = self.pool(out)
    x_161 = self.getattr_getattr_L__mod___stages___2_____1___self_attn_pool(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____1___post_attn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____1___post_attn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__22 = getattr_getattr_l__mod___stages___2_____1___post_attn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____1___post_attn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___post_attn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___post_attn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___post_attn_running_var = self.getattr_getattr_L__mod___stages___2_____1___post_attn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___post_attn_weight = self.getattr_getattr_L__mod___stages___2_____1___post_attn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___post_attn_bias = self.getattr_getattr_L__mod___stages___2_____1___post_attn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_162 = torch.nn.functional.batch_norm(x_161, getattr_getattr_l__mod___stages___2_____1___post_attn_running_mean, getattr_getattr_l__mod___stages___2_____1___post_attn_running_var, getattr_getattr_l__mod___stages___2_____1___post_attn_weight, getattr_getattr_l__mod___stages___2_____1___post_attn_bias, True, 0.1, 1e-05);  x_161 = getattr_getattr_l__mod___stages___2_____1___post_attn_running_mean = getattr_getattr_l__mod___stages___2_____1___post_attn_running_var = getattr_getattr_l__mod___stages___2_____1___post_attn_weight = getattr_getattr_l__mod___stages___2_____1___post_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_163 = self.getattr_getattr_L__mod___stages___2_____1___post_attn_drop(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_165 = self.getattr_getattr_L__mod___stages___2_____1___post_attn_act(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_166 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_conv(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__23 = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_167 = torch.nn.functional.batch_norm(x_166, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_166 = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___2_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_168 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_drop(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_171 = self.getattr_getattr_L__mod___stages___2_____1___conv3_1x1_bn_act(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:886, code: x = self.drop_path(x)
    x_172 = self.getattr_getattr_L__mod___stages___2_____1___drop_path(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___2_____1___shortcut = self.getattr_getattr_L__mod___stages___2_____1___shortcut(shortcut_5);  shortcut_5 = None
    x_173 = x_172 + getattr_getattr_l__mod___stages___2_____1___shortcut;  x_172 = getattr_getattr_l__mod___stages___2_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    shortcut_6 = self.getattr_getattr_L__mod___stages___2_____1___act(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_174 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_conv(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__24 = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_175 = torch.nn.functional.batch_norm(x_174, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_174 = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_176 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_drop(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_179 = self.getattr_getattr_L__mod___stages___3_____0___conv1_1x1_bn_act(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:882, code: x = self.conv2_kxk(x)
    x_180 = self.getattr_getattr_L__mod___stages___3_____0___conv2_kxk(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    q_5 = self.getattr_getattr_L__mod___stages___3_____0___self_attn_q(x_180)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    reshape_12 = q_5.reshape(-1, 16, 2, 4, 2, 4);  q_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    q_6 = reshape_12.permute(0, 1, 3, 5, 2, 4);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    reshape_13 = q_6.reshape(64, 16, -1, 4);  q_6 = None
    q_7 = reshape_13.transpose(1, 3);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    kv_3 = self.getattr_getattr_L__mod___stages___3_____0___self_attn_kv(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    kv_4 = torch.nn.functional.pad(kv_3, [2, 2, 2, 2]);  kv_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    unfold_2 = kv_4.unfold(2, 12, 8);  kv_4 = None
    unfold_3 = unfold_2.unfold(3, 12, 8);  unfold_2 = None
    reshape_14 = unfold_3.reshape(64, 80, 4, -1);  unfold_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    kv_5 = reshape_14.permute(0, 2, 3, 1);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    split_1 = torch.functional.split(kv_5, [16, 64], dim = -1);  kv_5 = None
    k_1 = split_1[0]
    v_1 = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    transpose_7 = k_1.transpose(-1, -2);  k_1 = None
    matmul_4 = q_7 @ transpose_7;  transpose_7 = None
    mul_6 = matmul_4 * 0.25;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    q_8 = q_7.reshape(-1, 4, 4, 16);  q_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:86, code: rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
    getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_width_rel = self.getattr_getattr_L__mod___stages___3_____0___self_attn_pos_embed_width_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_8 = getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_width_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_width_rel = None
    x_181 = q_8 @ transpose_8;  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    x_182 = x_181.reshape(-1, 4, 23);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_6 = torch.nn.functional.pad(x_182, [0, 1]);  x_182 = None
    x_pad_6 = pad_6.flatten(1);  pad_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    x_pad_7 = torch.nn.functional.pad(x_pad_6, [0, 19]);  x_pad_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x_pad_8 = x_pad_7.reshape(-1, 5, 23);  x_pad_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    x_183 = x_pad_8[(slice(None, None, None), slice(None, 4, None), slice(11, None, None))];  x_pad_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    reshape_18 = x_183.reshape(256, 4, 1, 4, 12);  x_183 = None
    x_184 = reshape_18.expand(-1, -1, 12, -1, -1);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    rel_logits_w_1 = x_184.permute((0, 1, 3, 2, 4));  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    q_9 = q_8.transpose(1, 2);  q_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:90, code: rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
    getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_height_rel = self.getattr_getattr_L__mod___stages___3_____0___self_attn_pos_embed_height_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_10 = getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_height_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____0___self_attn_pos_embed_height_rel = None
    x_185 = q_9 @ transpose_10;  q_9 = transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    x_186 = x_185.reshape(-1, 4, 23);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_8 = torch.nn.functional.pad(x_186, [0, 1]);  x_186 = None
    x_pad_9 = pad_8.flatten(1);  pad_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    x_pad_10 = torch.nn.functional.pad(x_pad_9, [0, 19]);  x_pad_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x_pad_11 = x_pad_10.reshape(-1, 5, 23);  x_pad_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    x_187 = x_pad_11[(slice(None, None, None), slice(None, 4, None), slice(11, None, None))];  x_pad_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    reshape_21 = x_187.reshape(256, 4, 1, 4, 12);  x_187 = None
    x_188 = reshape_21.expand(-1, -1, 12, -1, -1);  reshape_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    rel_logits_h_1 = x_188.permute((0, 3, 1, 4, 2));  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:92, code: rel_logits = rel_logits_h + rel_logits_w
    rel_logits_2 = rel_logits_h_1 + rel_logits_w_1;  rel_logits_h_1 = rel_logits_w_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    rel_logits_3 = rel_logits_2.reshape(64, 4, 16, -1);  rel_logits_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    attn_2 = mul_6 + rel_logits_3;  mul_6 = rel_logits_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    attn_3 = attn_2.softmax(dim = -1);  attn_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    matmul_7 = attn_3 @ v_1;  attn_3 = v_1 = None
    out_4 = matmul_7.transpose(1, 3);  matmul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    out_5 = out_4.reshape(-1, 4, 4, 2, 2);  out_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    permute_9 = out_5.permute(0, 3, 1, 4, 2);  out_5 = None
    contiguous_1 = permute_9.contiguous();  permute_9 = None
    out_6 = contiguous_1.view(8, 512, 8, 8);  contiguous_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:209, code: out = self.pool(out)
    x_189 = self.getattr_getattr_L__mod___stages___3_____0___self_attn_pool(out_6);  out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____0___post_attn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____0___post_attn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__25 = getattr_getattr_l__mod___stages___3_____0___post_attn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____0___post_attn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___post_attn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___post_attn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___post_attn_running_var = self.getattr_getattr_L__mod___stages___3_____0___post_attn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___post_attn_weight = self.getattr_getattr_L__mod___stages___3_____0___post_attn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___post_attn_bias = self.getattr_getattr_L__mod___stages___3_____0___post_attn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_190 = torch.nn.functional.batch_norm(x_189, getattr_getattr_l__mod___stages___3_____0___post_attn_running_mean, getattr_getattr_l__mod___stages___3_____0___post_attn_running_var, getattr_getattr_l__mod___stages___3_____0___post_attn_weight, getattr_getattr_l__mod___stages___3_____0___post_attn_bias, True, 0.1, 1e-05);  x_189 = getattr_getattr_l__mod___stages___3_____0___post_attn_running_mean = getattr_getattr_l__mod___stages___3_____0___post_attn_running_var = getattr_getattr_l__mod___stages___3_____0___post_attn_weight = getattr_getattr_l__mod___stages___3_____0___post_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_191 = self.getattr_getattr_L__mod___stages___3_____0___post_attn_drop(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_193 = self.getattr_getattr_L__mod___stages___3_____0___post_attn_act(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_194 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_conv(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__26 = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_195 = torch.nn.functional.batch_norm(x_194, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_194 = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____0___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_196 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_drop(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_199 = self.getattr_getattr_L__mod___stages___3_____0___conv3_1x1_bn_act(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:886, code: x = self.drop_path(x)
    x_200 = self.getattr_getattr_L__mod___stages___3_____0___drop_path(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_201 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_conv(shortcut_6);  shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__27 = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____0___shortcut_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_202 = torch.nn.functional.batch_norm(x_201, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight, getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias, True, 0.1, 1e-05);  x_201 = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_mean = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_running_var = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_weight = getattr_getattr_l__mod___stages___3_____0___shortcut_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_203 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_drop(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_205 = self.getattr_getattr_L__mod___stages___3_____0___shortcut_bn_act(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    x_206 = x_200 + x_205;  x_200 = x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    shortcut_7 = self.getattr_getattr_L__mod___stages___3_____0___act(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_207 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_conv(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__28 = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_208 = torch.nn.functional.batch_norm(x_207, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias, True, 0.1, 1e-05);  x_207 = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv1_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_209 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_drop(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_212 = self.getattr_getattr_L__mod___stages___3_____1___conv1_1x1_bn_act(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:882, code: x = self.conv2_kxk(x)
    x_213 = self.getattr_getattr_L__mod___stages___3_____1___conv2_kxk(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:177, code: q = self.q(x)
    q_10 = self.getattr_getattr_L__mod___stages___3_____1___self_attn_q(x_213)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:179, code: q = q.reshape(
    reshape_24 = q_10.reshape(-1, 16, 1, 8, 1, 8);  q_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:181, code: num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
    q_11 = reshape_24.permute(0, 1, 3, 5, 2, 4);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:183, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
    reshape_25 = q_11.reshape(64, 16, -1, 1);  q_11 = None
    q_12 = reshape_25.transpose(1, 3);  reshape_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:186, code: kv = self.kv(x)
    kv_6 = self.getattr_getattr_L__mod___stages___3_____1___self_attn_kv(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:190, code: kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
    kv_7 = torch.nn.functional.pad(kv_6, [2, 2, 2, 2]);  kv_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:191, code: kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
    unfold_4 = kv_7.unfold(2, 12, 8);  kv_7 = None
    unfold_5 = unfold_4.unfold(3, 12, 8);  unfold_4 = None
    reshape_26 = unfold_5.reshape(64, 80, 1, -1);  unfold_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:192, code: B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
    kv_8 = reshape_26.permute(0, 2, 3, 1);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:193, code: k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
    split_2 = torch.functional.split(kv_8, [16, 64], dim = -1);  kv_8 = None
    k_2 = split_2[0]
    v_2 = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    transpose_13 = k_2.transpose(-1, -2);  k_2 = None
    matmul_8 = q_12 @ transpose_13;  transpose_13 = None
    mul_7 = matmul_8 * 0.25;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:85, code: q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
    q_13 = q_12.reshape(-1, 8, 8, 16);  q_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:86, code: rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
    getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_width_rel = self.getattr_getattr_L__mod___stages___3_____1___self_attn_pos_embed_width_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_14 = getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_width_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_width_rel = None
    x_214 = q_13 @ transpose_14;  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    x_215 = x_214.reshape(-1, 8, 23);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_11 = torch.nn.functional.pad(x_215, [0, 1]);  x_215 = None
    x_pad_12 = pad_11.flatten(1);  pad_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    x_pad_13 = torch.nn.functional.pad(x_pad_12, [0, 15]);  x_pad_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x_pad_14 = x_pad_13.reshape(-1, 9, 23);  x_pad_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    x_216 = x_pad_14[(slice(None, None, None), slice(None, 8, None), slice(11, None, None))];  x_pad_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    reshape_30 = x_216.reshape(64, 8, 1, 8, 12);  x_216 = None
    x_217 = reshape_30.expand(-1, -1, 12, -1, -1);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    rel_logits_w_2 = x_217.permute((0, 1, 3, 2, 4));  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:89, code: q = q.transpose(1, 2)
    q_14 = q_13.transpose(1, 2);  q_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:90, code: rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
    getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_height_rel = self.getattr_getattr_L__mod___stages___3_____1___self_attn_pos_embed_height_rel
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:45, code: x = (q @ rel_k.transpose(-1, -2))
    transpose_16 = getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_height_rel.transpose(-1, -2);  getattr_getattr_l__mod___stages___3_____1___self_attn_pos_embed_height_rel = None
    x_218 = q_14 @ transpose_16;  q_14 = transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:46, code: x = x.reshape(-1, W, rel_size)
    x_219 = x_218.reshape(-1, 8, 23);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:49, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    pad_13 = torch.nn.functional.pad(x_219, [0, 1]);  x_219 = None
    x_pad_15 = pad_13.flatten(1);  pad_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:50, code: x_pad = F.pad(x_pad, [0, rel_size - W])
    x_pad_16 = torch.nn.functional.pad(x_pad_15, [0, 15]);  x_pad_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:53, code: x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x_pad_17 = x_pad_16.reshape(-1, 9, 23);  x_pad_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:54, code: x = x_pad[:, :W, win_size - 1:]
    x_220 = x_pad_17[(slice(None, None, None), slice(None, 8, None), slice(11, None, None))];  x_pad_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:57, code: x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    reshape_33 = x_220.reshape(64, 8, 1, 8, 12);  x_220 = None
    x_221 = reshape_33.expand(-1, -1, 12, -1, -1);  reshape_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:58, code: return x.permute(permute_mask)
    rel_logits_h_2 = x_221.permute((0, 3, 1, 4, 2));  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:92, code: rel_logits = rel_logits_h + rel_logits_w
    rel_logits_4 = rel_logits_h_2 + rel_logits_w_2;  rel_logits_h_2 = rel_logits_w_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:93, code: rel_logits = rel_logits.reshape(B, BB, HW, -1)
    rel_logits_5 = rel_logits_4.reshape(64, 1, 64, -1);  rel_logits_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:199, code: attn = (q @ k.transpose(-1, -2)) * self.scale + self.pos_embed(q)
    attn_4 = mul_7 + rel_logits_5;  mul_7 = rel_logits_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:201, code: attn = attn.softmax(dim=-1)
    attn_5 = attn_4.softmax(dim = -1);  attn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:203, code: out = (attn @ v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
    matmul_11 = attn_5 @ v_2;  attn_5 = v_2 = None
    out_8 = matmul_11.transpose(1, 3);  matmul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:205, code: out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
    out_9 = out_8.reshape(-1, 8, 8, 1, 1);  out_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:206, code: out = out.permute(0, 3, 1, 4, 2).contiguous().view(
    permute_14 = out_9.permute(0, 3, 1, 4, 2);  out_9 = None
    contiguous_2 = permute_14.contiguous();  permute_14 = None
    out_10 = contiguous_2.view(8, 512, 8, 8);  contiguous_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/halo_attn.py:209, code: out = self.pool(out)
    x_222 = self.getattr_getattr_L__mod___stages___3_____1___self_attn_pool(out_10);  out_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____1___post_attn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____1___post_attn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__29 = getattr_getattr_l__mod___stages___3_____1___post_attn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____1___post_attn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___post_attn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___post_attn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___post_attn_running_var = self.getattr_getattr_L__mod___stages___3_____1___post_attn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___post_attn_weight = self.getattr_getattr_L__mod___stages___3_____1___post_attn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___post_attn_bias = self.getattr_getattr_L__mod___stages___3_____1___post_attn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_223 = torch.nn.functional.batch_norm(x_222, getattr_getattr_l__mod___stages___3_____1___post_attn_running_mean, getattr_getattr_l__mod___stages___3_____1___post_attn_running_var, getattr_getattr_l__mod___stages___3_____1___post_attn_weight, getattr_getattr_l__mod___stages___3_____1___post_attn_bias, True, 0.1, 1e-05);  x_222 = getattr_getattr_l__mod___stages___3_____1___post_attn_running_mean = getattr_getattr_l__mod___stages___3_____1___post_attn_running_var = getattr_getattr_l__mod___stages___3_____1___post_attn_weight = getattr_getattr_l__mod___stages___3_____1___post_attn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_224 = self.getattr_getattr_L__mod___stages___3_____1___post_attn_drop(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_226 = self.getattr_getattr_L__mod___stages___3_____1___post_attn_act(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_227 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_conv(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_num_batches_tracked = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__30 = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_num_batches_tracked.add_(1);  getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight, getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias, True, 0.1, 1e-05);  x_227 = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_mean = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_running_var = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_weight = getattr_getattr_l__mod___stages___3_____1___conv3_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_232 = self.getattr_getattr_L__mod___stages___3_____1___conv3_1x1_bn_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:886, code: x = self.drop_path(x)
    x_233 = self.getattr_getattr_L__mod___stages___3_____1___drop_path(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    getattr_getattr_l__mod___stages___3_____1___shortcut = self.getattr_getattr_L__mod___stages___3_____1___shortcut(shortcut_7);  shortcut_7 = None
    x_234 = x_233 + getattr_getattr_l__mod___stages___3_____1___shortcut;  x_233 = getattr_getattr_l__mod___stages___3_____1___shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    x_235 = self.getattr_getattr_L__mod___stages___3_____1___act(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1250, code: x = self.final_conv(x)
    x_237 = self.L__mod___final_conv(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_238 = self.L__mod___head_global_pool_pool(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_240 = self.L__mod___head_global_pool_flatten(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    x_241 = self.L__mod___head_drop(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    x_242 = self.L__mod___head_fc(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:135, code: return self.flatten(x)
    pred = self.L__mod___head_flatten(x_242);  x_242 = None
    return (pred,)
    