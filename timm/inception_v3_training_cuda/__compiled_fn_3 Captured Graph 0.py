from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x = self.L__mod___Conv2d_1a_3x3_conv(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___conv2d_1a_3x3_bn_num_batches_tracked = self.L__mod___Conv2d_1a_3x3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_ = l__mod___conv2d_1a_3x3_bn_num_batches_tracked.add_(1);  l__mod___conv2d_1a_3x3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___conv2d_1a_3x3_bn_running_mean = self.L__mod___Conv2d_1a_3x3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___conv2d_1a_3x3_bn_running_var = self.L__mod___Conv2d_1a_3x3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___conv2d_1a_3x3_bn_weight = self.L__mod___Conv2d_1a_3x3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___conv2d_1a_3x3_bn_bias = self.L__mod___Conv2d_1a_3x3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___conv2d_1a_3x3_bn_running_mean, l__mod___conv2d_1a_3x3_bn_running_var, l__mod___conv2d_1a_3x3_bn_weight, l__mod___conv2d_1a_3x3_bn_bias, True, 0.1, 0.001);  x = l__mod___conv2d_1a_3x3_bn_running_mean = l__mod___conv2d_1a_3x3_bn_running_var = l__mod___conv2d_1a_3x3_bn_weight = l__mod___conv2d_1a_3x3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___Conv2d_1a_3x3_bn_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_5 = self.L__mod___Conv2d_1a_3x3_bn_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_6 = self.L__mod___Conv2d_2a_3x3_conv(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___conv2d_2a_3x3_bn_num_batches_tracked = self.L__mod___Conv2d_2a_3x3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__1 = l__mod___conv2d_2a_3x3_bn_num_batches_tracked.add_(1);  l__mod___conv2d_2a_3x3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___conv2d_2a_3x3_bn_running_mean = self.L__mod___Conv2d_2a_3x3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___conv2d_2a_3x3_bn_running_var = self.L__mod___Conv2d_2a_3x3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___conv2d_2a_3x3_bn_weight = self.L__mod___Conv2d_2a_3x3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___conv2d_2a_3x3_bn_bias = self.L__mod___Conv2d_2a_3x3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_7 = torch.nn.functional.batch_norm(x_6, l__mod___conv2d_2a_3x3_bn_running_mean, l__mod___conv2d_2a_3x3_bn_running_var, l__mod___conv2d_2a_3x3_bn_weight, l__mod___conv2d_2a_3x3_bn_bias, True, 0.1, 0.001);  x_6 = l__mod___conv2d_2a_3x3_bn_running_mean = l__mod___conv2d_2a_3x3_bn_running_var = l__mod___conv2d_2a_3x3_bn_weight = l__mod___conv2d_2a_3x3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_8 = self.L__mod___Conv2d_2a_3x3_bn_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_11 = self.L__mod___Conv2d_2a_3x3_bn_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_12 = self.L__mod___Conv2d_2b_3x3_conv(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___conv2d_2b_3x3_bn_num_batches_tracked = self.L__mod___Conv2d_2b_3x3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__2 = l__mod___conv2d_2b_3x3_bn_num_batches_tracked.add_(1);  l__mod___conv2d_2b_3x3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___conv2d_2b_3x3_bn_running_mean = self.L__mod___Conv2d_2b_3x3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___conv2d_2b_3x3_bn_running_var = self.L__mod___Conv2d_2b_3x3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___conv2d_2b_3x3_bn_weight = self.L__mod___Conv2d_2b_3x3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___conv2d_2b_3x3_bn_bias = self.L__mod___Conv2d_2b_3x3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_13 = torch.nn.functional.batch_norm(x_12, l__mod___conv2d_2b_3x3_bn_running_mean, l__mod___conv2d_2b_3x3_bn_running_var, l__mod___conv2d_2b_3x3_bn_weight, l__mod___conv2d_2b_3x3_bn_bias, True, 0.1, 0.001);  x_12 = l__mod___conv2d_2b_3x3_bn_running_mean = l__mod___conv2d_2b_3x3_bn_running_var = l__mod___conv2d_2b_3x3_bn_weight = l__mod___conv2d_2b_3x3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_14 = self.L__mod___Conv2d_2b_3x3_bn_drop(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_17 = self.L__mod___Conv2d_2b_3x3_bn_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:345, code: x = self.Pool1(x)  # N x 64 x 73 x 73
    x_18 = self.L__mod___Pool1(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_19 = self.L__mod___Conv2d_3b_1x1_conv(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___conv2d_3b_1x1_bn_num_batches_tracked = self.L__mod___Conv2d_3b_1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__3 = l__mod___conv2d_3b_1x1_bn_num_batches_tracked.add_(1);  l__mod___conv2d_3b_1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___conv2d_3b_1x1_bn_running_mean = self.L__mod___Conv2d_3b_1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___conv2d_3b_1x1_bn_running_var = self.L__mod___Conv2d_3b_1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___conv2d_3b_1x1_bn_weight = self.L__mod___Conv2d_3b_1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___conv2d_3b_1x1_bn_bias = self.L__mod___Conv2d_3b_1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_20 = torch.nn.functional.batch_norm(x_19, l__mod___conv2d_3b_1x1_bn_running_mean, l__mod___conv2d_3b_1x1_bn_running_var, l__mod___conv2d_3b_1x1_bn_weight, l__mod___conv2d_3b_1x1_bn_bias, True, 0.1, 0.001);  x_19 = l__mod___conv2d_3b_1x1_bn_running_mean = l__mod___conv2d_3b_1x1_bn_running_var = l__mod___conv2d_3b_1x1_bn_weight = l__mod___conv2d_3b_1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_21 = self.L__mod___Conv2d_3b_1x1_bn_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_24 = self.L__mod___Conv2d_3b_1x1_bn_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_25 = self.L__mod___Conv2d_4a_3x3_conv(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___conv2d_4a_3x3_bn_num_batches_tracked = self.L__mod___Conv2d_4a_3x3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__4 = l__mod___conv2d_4a_3x3_bn_num_batches_tracked.add_(1);  l__mod___conv2d_4a_3x3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___conv2d_4a_3x3_bn_running_mean = self.L__mod___Conv2d_4a_3x3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___conv2d_4a_3x3_bn_running_var = self.L__mod___Conv2d_4a_3x3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___conv2d_4a_3x3_bn_weight = self.L__mod___Conv2d_4a_3x3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___conv2d_4a_3x3_bn_bias = self.L__mod___Conv2d_4a_3x3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_26 = torch.nn.functional.batch_norm(x_25, l__mod___conv2d_4a_3x3_bn_running_mean, l__mod___conv2d_4a_3x3_bn_running_var, l__mod___conv2d_4a_3x3_bn_weight, l__mod___conv2d_4a_3x3_bn_bias, True, 0.1, 0.001);  x_25 = l__mod___conv2d_4a_3x3_bn_running_mean = l__mod___conv2d_4a_3x3_bn_running_var = l__mod___conv2d_4a_3x3_bn_weight = l__mod___conv2d_4a_3x3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_27 = self.L__mod___Conv2d_4a_3x3_bn_drop(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_30 = self.L__mod___Conv2d_4a_3x3_bn_act(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:348, code: x = self.Pool2(x)  # N x 192 x 35 x 35
    x_31 = self.L__mod___Pool2(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_32 = self.L__mod___Mixed_5b_branch1x1_conv(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5b_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_5b_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__5 = l__mod___mixed_5b_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5b_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch1x1_bn_running_mean = self.L__mod___Mixed_5b_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch1x1_bn_running_var = self.L__mod___Mixed_5b_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5b_branch1x1_bn_weight = self.L__mod___Mixed_5b_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5b_branch1x1_bn_bias = self.L__mod___Mixed_5b_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_33 = torch.nn.functional.batch_norm(x_32, l__mod___mixed_5b_branch1x1_bn_running_mean, l__mod___mixed_5b_branch1x1_bn_running_var, l__mod___mixed_5b_branch1x1_bn_weight, l__mod___mixed_5b_branch1x1_bn_bias, True, 0.1, 0.001);  x_32 = l__mod___mixed_5b_branch1x1_bn_running_mean = l__mod___mixed_5b_branch1x1_bn_running_var = l__mod___mixed_5b_branch1x1_bn_weight = l__mod___mixed_5b_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_34 = self.L__mod___Mixed_5b_branch1x1_bn_drop(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1 = self.L__mod___Mixed_5b_branch1x1_bn_act(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_37 = self.L__mod___Mixed_5b_branch5x5_1_conv(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5b_branch5x5_1_bn_num_batches_tracked = self.L__mod___Mixed_5b_branch5x5_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__6 = l__mod___mixed_5b_branch5x5_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5b_branch5x5_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch5x5_1_bn_running_mean = self.L__mod___Mixed_5b_branch5x5_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch5x5_1_bn_running_var = self.L__mod___Mixed_5b_branch5x5_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5b_branch5x5_1_bn_weight = self.L__mod___Mixed_5b_branch5x5_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5b_branch5x5_1_bn_bias = self.L__mod___Mixed_5b_branch5x5_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_38 = torch.nn.functional.batch_norm(x_37, l__mod___mixed_5b_branch5x5_1_bn_running_mean, l__mod___mixed_5b_branch5x5_1_bn_running_var, l__mod___mixed_5b_branch5x5_1_bn_weight, l__mod___mixed_5b_branch5x5_1_bn_bias, True, 0.1, 0.001);  x_37 = l__mod___mixed_5b_branch5x5_1_bn_running_mean = l__mod___mixed_5b_branch5x5_1_bn_running_var = l__mod___mixed_5b_branch5x5_1_bn_weight = l__mod___mixed_5b_branch5x5_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_39 = self.L__mod___Mixed_5b_branch5x5_1_bn_drop(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch5x5 = self.L__mod___Mixed_5b_branch5x5_1_bn_act(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_42 = self.L__mod___Mixed_5b_branch5x5_2_conv(branch5x5);  branch5x5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5b_branch5x5_2_bn_num_batches_tracked = self.L__mod___Mixed_5b_branch5x5_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__7 = l__mod___mixed_5b_branch5x5_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_5b_branch5x5_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch5x5_2_bn_running_mean = self.L__mod___Mixed_5b_branch5x5_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch5x5_2_bn_running_var = self.L__mod___Mixed_5b_branch5x5_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5b_branch5x5_2_bn_weight = self.L__mod___Mixed_5b_branch5x5_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5b_branch5x5_2_bn_bias = self.L__mod___Mixed_5b_branch5x5_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_43 = torch.nn.functional.batch_norm(x_42, l__mod___mixed_5b_branch5x5_2_bn_running_mean, l__mod___mixed_5b_branch5x5_2_bn_running_var, l__mod___mixed_5b_branch5x5_2_bn_weight, l__mod___mixed_5b_branch5x5_2_bn_bias, True, 0.1, 0.001);  x_42 = l__mod___mixed_5b_branch5x5_2_bn_running_mean = l__mod___mixed_5b_branch5x5_2_bn_running_var = l__mod___mixed_5b_branch5x5_2_bn_weight = l__mod___mixed_5b_branch5x5_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_44 = self.L__mod___Mixed_5b_branch5x5_2_bn_drop(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch5x5_1 = self.L__mod___Mixed_5b_branch5x5_2_bn_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_47 = self.L__mod___Mixed_5b_branch3x3dbl_1_conv(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5b_branch3x3dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_5b_branch3x3dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__8 = l__mod___mixed_5b_branch3x3dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5b_branch3x3dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch3x3dbl_1_bn_running_mean = self.L__mod___Mixed_5b_branch3x3dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch3x3dbl_1_bn_running_var = self.L__mod___Mixed_5b_branch3x3dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5b_branch3x3dbl_1_bn_weight = self.L__mod___Mixed_5b_branch3x3dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5b_branch3x3dbl_1_bn_bias = self.L__mod___Mixed_5b_branch3x3dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_48 = torch.nn.functional.batch_norm(x_47, l__mod___mixed_5b_branch3x3dbl_1_bn_running_mean, l__mod___mixed_5b_branch3x3dbl_1_bn_running_var, l__mod___mixed_5b_branch3x3dbl_1_bn_weight, l__mod___mixed_5b_branch3x3dbl_1_bn_bias, True, 0.1, 0.001);  x_47 = l__mod___mixed_5b_branch3x3dbl_1_bn_running_mean = l__mod___mixed_5b_branch3x3dbl_1_bn_running_var = l__mod___mixed_5b_branch3x3dbl_1_bn_weight = l__mod___mixed_5b_branch3x3dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_49 = self.L__mod___Mixed_5b_branch3x3dbl_1_bn_drop(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl = self.L__mod___Mixed_5b_branch3x3dbl_1_bn_act(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_52 = self.L__mod___Mixed_5b_branch3x3dbl_2_conv(branch3x3dbl);  branch3x3dbl = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5b_branch3x3dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_5b_branch3x3dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__9 = l__mod___mixed_5b_branch3x3dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_5b_branch3x3dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch3x3dbl_2_bn_running_mean = self.L__mod___Mixed_5b_branch3x3dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch3x3dbl_2_bn_running_var = self.L__mod___Mixed_5b_branch3x3dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5b_branch3x3dbl_2_bn_weight = self.L__mod___Mixed_5b_branch3x3dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5b_branch3x3dbl_2_bn_bias = self.L__mod___Mixed_5b_branch3x3dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_53 = torch.nn.functional.batch_norm(x_52, l__mod___mixed_5b_branch3x3dbl_2_bn_running_mean, l__mod___mixed_5b_branch3x3dbl_2_bn_running_var, l__mod___mixed_5b_branch3x3dbl_2_bn_weight, l__mod___mixed_5b_branch3x3dbl_2_bn_bias, True, 0.1, 0.001);  x_52 = l__mod___mixed_5b_branch3x3dbl_2_bn_running_mean = l__mod___mixed_5b_branch3x3dbl_2_bn_running_var = l__mod___mixed_5b_branch3x3dbl_2_bn_weight = l__mod___mixed_5b_branch3x3dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_54 = self.L__mod___Mixed_5b_branch3x3dbl_2_bn_drop(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_1 = self.L__mod___Mixed_5b_branch3x3dbl_2_bn_act(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_57 = self.L__mod___Mixed_5b_branch3x3dbl_3_conv(branch3x3dbl_1);  branch3x3dbl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5b_branch3x3dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_5b_branch3x3dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__10 = l__mod___mixed_5b_branch3x3dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_5b_branch3x3dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch3x3dbl_3_bn_running_mean = self.L__mod___Mixed_5b_branch3x3dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch3x3dbl_3_bn_running_var = self.L__mod___Mixed_5b_branch3x3dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5b_branch3x3dbl_3_bn_weight = self.L__mod___Mixed_5b_branch3x3dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5b_branch3x3dbl_3_bn_bias = self.L__mod___Mixed_5b_branch3x3dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_58 = torch.nn.functional.batch_norm(x_57, l__mod___mixed_5b_branch3x3dbl_3_bn_running_mean, l__mod___mixed_5b_branch3x3dbl_3_bn_running_var, l__mod___mixed_5b_branch3x3dbl_3_bn_weight, l__mod___mixed_5b_branch3x3dbl_3_bn_bias, True, 0.1, 0.001);  x_57 = l__mod___mixed_5b_branch3x3dbl_3_bn_running_mean = l__mod___mixed_5b_branch3x3dbl_3_bn_running_var = l__mod___mixed_5b_branch3x3dbl_3_bn_weight = l__mod___mixed_5b_branch3x3dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_59 = self.L__mod___Mixed_5b_branch3x3dbl_3_bn_drop(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_2 = self.L__mod___Mixed_5b_branch3x3dbl_3_bn_act(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = torch._C._nn.avg_pool2d(x_31, kernel_size = 3, stride = 1, padding = 1);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_62 = self.L__mod___Mixed_5b_branch_pool_conv(branch_pool);  branch_pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5b_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_5b_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__11 = l__mod___mixed_5b_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_5b_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch_pool_bn_running_mean = self.L__mod___Mixed_5b_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5b_branch_pool_bn_running_var = self.L__mod___Mixed_5b_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5b_branch_pool_bn_weight = self.L__mod___Mixed_5b_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5b_branch_pool_bn_bias = self.L__mod___Mixed_5b_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_63 = torch.nn.functional.batch_norm(x_62, l__mod___mixed_5b_branch_pool_bn_running_mean, l__mod___mixed_5b_branch_pool_bn_running_var, l__mod___mixed_5b_branch_pool_bn_weight, l__mod___mixed_5b_branch_pool_bn_bias, True, 0.1, 0.001);  x_62 = l__mod___mixed_5b_branch_pool_bn_running_mean = l__mod___mixed_5b_branch_pool_bn_running_var = l__mod___mixed_5b_branch_pool_bn_weight = l__mod___mixed_5b_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_64 = self.L__mod___Mixed_5b_branch_pool_bn_drop(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_1 = self.L__mod___Mixed_5b_branch_pool_bn_act(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    x_67 = torch.cat([branch1x1, branch5x5_1, branch3x3dbl_2, branch_pool_1], 1);  branch1x1 = branch5x5_1 = branch3x3dbl_2 = branch_pool_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_68 = self.L__mod___Mixed_5c_branch1x1_conv(x_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5c_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_5c_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__12 = l__mod___mixed_5c_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5c_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch1x1_bn_running_mean = self.L__mod___Mixed_5c_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch1x1_bn_running_var = self.L__mod___Mixed_5c_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5c_branch1x1_bn_weight = self.L__mod___Mixed_5c_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5c_branch1x1_bn_bias = self.L__mod___Mixed_5c_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_69 = torch.nn.functional.batch_norm(x_68, l__mod___mixed_5c_branch1x1_bn_running_mean, l__mod___mixed_5c_branch1x1_bn_running_var, l__mod___mixed_5c_branch1x1_bn_weight, l__mod___mixed_5c_branch1x1_bn_bias, True, 0.1, 0.001);  x_68 = l__mod___mixed_5c_branch1x1_bn_running_mean = l__mod___mixed_5c_branch1x1_bn_running_var = l__mod___mixed_5c_branch1x1_bn_weight = l__mod___mixed_5c_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_70 = self.L__mod___Mixed_5c_branch1x1_bn_drop(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_1 = self.L__mod___Mixed_5c_branch1x1_bn_act(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_73 = self.L__mod___Mixed_5c_branch5x5_1_conv(x_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5c_branch5x5_1_bn_num_batches_tracked = self.L__mod___Mixed_5c_branch5x5_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__13 = l__mod___mixed_5c_branch5x5_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5c_branch5x5_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch5x5_1_bn_running_mean = self.L__mod___Mixed_5c_branch5x5_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch5x5_1_bn_running_var = self.L__mod___Mixed_5c_branch5x5_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5c_branch5x5_1_bn_weight = self.L__mod___Mixed_5c_branch5x5_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5c_branch5x5_1_bn_bias = self.L__mod___Mixed_5c_branch5x5_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_74 = torch.nn.functional.batch_norm(x_73, l__mod___mixed_5c_branch5x5_1_bn_running_mean, l__mod___mixed_5c_branch5x5_1_bn_running_var, l__mod___mixed_5c_branch5x5_1_bn_weight, l__mod___mixed_5c_branch5x5_1_bn_bias, True, 0.1, 0.001);  x_73 = l__mod___mixed_5c_branch5x5_1_bn_running_mean = l__mod___mixed_5c_branch5x5_1_bn_running_var = l__mod___mixed_5c_branch5x5_1_bn_weight = l__mod___mixed_5c_branch5x5_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_75 = self.L__mod___Mixed_5c_branch5x5_1_bn_drop(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch5x5_2 = self.L__mod___Mixed_5c_branch5x5_1_bn_act(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_78 = self.L__mod___Mixed_5c_branch5x5_2_conv(branch5x5_2);  branch5x5_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5c_branch5x5_2_bn_num_batches_tracked = self.L__mod___Mixed_5c_branch5x5_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__14 = l__mod___mixed_5c_branch5x5_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_5c_branch5x5_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch5x5_2_bn_running_mean = self.L__mod___Mixed_5c_branch5x5_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch5x5_2_bn_running_var = self.L__mod___Mixed_5c_branch5x5_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5c_branch5x5_2_bn_weight = self.L__mod___Mixed_5c_branch5x5_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5c_branch5x5_2_bn_bias = self.L__mod___Mixed_5c_branch5x5_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_79 = torch.nn.functional.batch_norm(x_78, l__mod___mixed_5c_branch5x5_2_bn_running_mean, l__mod___mixed_5c_branch5x5_2_bn_running_var, l__mod___mixed_5c_branch5x5_2_bn_weight, l__mod___mixed_5c_branch5x5_2_bn_bias, True, 0.1, 0.001);  x_78 = l__mod___mixed_5c_branch5x5_2_bn_running_mean = l__mod___mixed_5c_branch5x5_2_bn_running_var = l__mod___mixed_5c_branch5x5_2_bn_weight = l__mod___mixed_5c_branch5x5_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_80 = self.L__mod___Mixed_5c_branch5x5_2_bn_drop(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch5x5_3 = self.L__mod___Mixed_5c_branch5x5_2_bn_act(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_83 = self.L__mod___Mixed_5c_branch3x3dbl_1_conv(x_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5c_branch3x3dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_5c_branch3x3dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__15 = l__mod___mixed_5c_branch3x3dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5c_branch3x3dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch3x3dbl_1_bn_running_mean = self.L__mod___Mixed_5c_branch3x3dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch3x3dbl_1_bn_running_var = self.L__mod___Mixed_5c_branch3x3dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5c_branch3x3dbl_1_bn_weight = self.L__mod___Mixed_5c_branch3x3dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5c_branch3x3dbl_1_bn_bias = self.L__mod___Mixed_5c_branch3x3dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_84 = torch.nn.functional.batch_norm(x_83, l__mod___mixed_5c_branch3x3dbl_1_bn_running_mean, l__mod___mixed_5c_branch3x3dbl_1_bn_running_var, l__mod___mixed_5c_branch3x3dbl_1_bn_weight, l__mod___mixed_5c_branch3x3dbl_1_bn_bias, True, 0.1, 0.001);  x_83 = l__mod___mixed_5c_branch3x3dbl_1_bn_running_mean = l__mod___mixed_5c_branch3x3dbl_1_bn_running_var = l__mod___mixed_5c_branch3x3dbl_1_bn_weight = l__mod___mixed_5c_branch3x3dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_85 = self.L__mod___Mixed_5c_branch3x3dbl_1_bn_drop(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_3 = self.L__mod___Mixed_5c_branch3x3dbl_1_bn_act(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_88 = self.L__mod___Mixed_5c_branch3x3dbl_2_conv(branch3x3dbl_3);  branch3x3dbl_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5c_branch3x3dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_5c_branch3x3dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__16 = l__mod___mixed_5c_branch3x3dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_5c_branch3x3dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch3x3dbl_2_bn_running_mean = self.L__mod___Mixed_5c_branch3x3dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch3x3dbl_2_bn_running_var = self.L__mod___Mixed_5c_branch3x3dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5c_branch3x3dbl_2_bn_weight = self.L__mod___Mixed_5c_branch3x3dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5c_branch3x3dbl_2_bn_bias = self.L__mod___Mixed_5c_branch3x3dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_89 = torch.nn.functional.batch_norm(x_88, l__mod___mixed_5c_branch3x3dbl_2_bn_running_mean, l__mod___mixed_5c_branch3x3dbl_2_bn_running_var, l__mod___mixed_5c_branch3x3dbl_2_bn_weight, l__mod___mixed_5c_branch3x3dbl_2_bn_bias, True, 0.1, 0.001);  x_88 = l__mod___mixed_5c_branch3x3dbl_2_bn_running_mean = l__mod___mixed_5c_branch3x3dbl_2_bn_running_var = l__mod___mixed_5c_branch3x3dbl_2_bn_weight = l__mod___mixed_5c_branch3x3dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_90 = self.L__mod___Mixed_5c_branch3x3dbl_2_bn_drop(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_4 = self.L__mod___Mixed_5c_branch3x3dbl_2_bn_act(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_93 = self.L__mod___Mixed_5c_branch3x3dbl_3_conv(branch3x3dbl_4);  branch3x3dbl_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5c_branch3x3dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_5c_branch3x3dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__17 = l__mod___mixed_5c_branch3x3dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_5c_branch3x3dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch3x3dbl_3_bn_running_mean = self.L__mod___Mixed_5c_branch3x3dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch3x3dbl_3_bn_running_var = self.L__mod___Mixed_5c_branch3x3dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5c_branch3x3dbl_3_bn_weight = self.L__mod___Mixed_5c_branch3x3dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5c_branch3x3dbl_3_bn_bias = self.L__mod___Mixed_5c_branch3x3dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_94 = torch.nn.functional.batch_norm(x_93, l__mod___mixed_5c_branch3x3dbl_3_bn_running_mean, l__mod___mixed_5c_branch3x3dbl_3_bn_running_var, l__mod___mixed_5c_branch3x3dbl_3_bn_weight, l__mod___mixed_5c_branch3x3dbl_3_bn_bias, True, 0.1, 0.001);  x_93 = l__mod___mixed_5c_branch3x3dbl_3_bn_running_mean = l__mod___mixed_5c_branch3x3dbl_3_bn_running_var = l__mod___mixed_5c_branch3x3dbl_3_bn_weight = l__mod___mixed_5c_branch3x3dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_95 = self.L__mod___Mixed_5c_branch3x3dbl_3_bn_drop(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_5 = self.L__mod___Mixed_5c_branch3x3dbl_3_bn_act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_2 = torch._C._nn.avg_pool2d(x_67, kernel_size = 3, stride = 1, padding = 1);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_98 = self.L__mod___Mixed_5c_branch_pool_conv(branch_pool_2);  branch_pool_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5c_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_5c_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__18 = l__mod___mixed_5c_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_5c_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch_pool_bn_running_mean = self.L__mod___Mixed_5c_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5c_branch_pool_bn_running_var = self.L__mod___Mixed_5c_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5c_branch_pool_bn_weight = self.L__mod___Mixed_5c_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5c_branch_pool_bn_bias = self.L__mod___Mixed_5c_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_99 = torch.nn.functional.batch_norm(x_98, l__mod___mixed_5c_branch_pool_bn_running_mean, l__mod___mixed_5c_branch_pool_bn_running_var, l__mod___mixed_5c_branch_pool_bn_weight, l__mod___mixed_5c_branch_pool_bn_bias, True, 0.1, 0.001);  x_98 = l__mod___mixed_5c_branch_pool_bn_running_mean = l__mod___mixed_5c_branch_pool_bn_running_var = l__mod___mixed_5c_branch_pool_bn_weight = l__mod___mixed_5c_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_100 = self.L__mod___Mixed_5c_branch_pool_bn_drop(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_3 = self.L__mod___Mixed_5c_branch_pool_bn_act(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    x_103 = torch.cat([branch1x1_1, branch5x5_3, branch3x3dbl_5, branch_pool_3], 1);  branch1x1_1 = branch5x5_3 = branch3x3dbl_5 = branch_pool_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_104 = self.L__mod___Mixed_5d_branch1x1_conv(x_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5d_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_5d_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__19 = l__mod___mixed_5d_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5d_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch1x1_bn_running_mean = self.L__mod___Mixed_5d_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch1x1_bn_running_var = self.L__mod___Mixed_5d_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5d_branch1x1_bn_weight = self.L__mod___Mixed_5d_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5d_branch1x1_bn_bias = self.L__mod___Mixed_5d_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_105 = torch.nn.functional.batch_norm(x_104, l__mod___mixed_5d_branch1x1_bn_running_mean, l__mod___mixed_5d_branch1x1_bn_running_var, l__mod___mixed_5d_branch1x1_bn_weight, l__mod___mixed_5d_branch1x1_bn_bias, True, 0.1, 0.001);  x_104 = l__mod___mixed_5d_branch1x1_bn_running_mean = l__mod___mixed_5d_branch1x1_bn_running_var = l__mod___mixed_5d_branch1x1_bn_weight = l__mod___mixed_5d_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_106 = self.L__mod___Mixed_5d_branch1x1_bn_drop(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_2 = self.L__mod___Mixed_5d_branch1x1_bn_act(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_109 = self.L__mod___Mixed_5d_branch5x5_1_conv(x_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5d_branch5x5_1_bn_num_batches_tracked = self.L__mod___Mixed_5d_branch5x5_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__20 = l__mod___mixed_5d_branch5x5_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5d_branch5x5_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch5x5_1_bn_running_mean = self.L__mod___Mixed_5d_branch5x5_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch5x5_1_bn_running_var = self.L__mod___Mixed_5d_branch5x5_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5d_branch5x5_1_bn_weight = self.L__mod___Mixed_5d_branch5x5_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5d_branch5x5_1_bn_bias = self.L__mod___Mixed_5d_branch5x5_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_110 = torch.nn.functional.batch_norm(x_109, l__mod___mixed_5d_branch5x5_1_bn_running_mean, l__mod___mixed_5d_branch5x5_1_bn_running_var, l__mod___mixed_5d_branch5x5_1_bn_weight, l__mod___mixed_5d_branch5x5_1_bn_bias, True, 0.1, 0.001);  x_109 = l__mod___mixed_5d_branch5x5_1_bn_running_mean = l__mod___mixed_5d_branch5x5_1_bn_running_var = l__mod___mixed_5d_branch5x5_1_bn_weight = l__mod___mixed_5d_branch5x5_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_111 = self.L__mod___Mixed_5d_branch5x5_1_bn_drop(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch5x5_4 = self.L__mod___Mixed_5d_branch5x5_1_bn_act(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_114 = self.L__mod___Mixed_5d_branch5x5_2_conv(branch5x5_4);  branch5x5_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5d_branch5x5_2_bn_num_batches_tracked = self.L__mod___Mixed_5d_branch5x5_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__21 = l__mod___mixed_5d_branch5x5_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_5d_branch5x5_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch5x5_2_bn_running_mean = self.L__mod___Mixed_5d_branch5x5_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch5x5_2_bn_running_var = self.L__mod___Mixed_5d_branch5x5_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5d_branch5x5_2_bn_weight = self.L__mod___Mixed_5d_branch5x5_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5d_branch5x5_2_bn_bias = self.L__mod___Mixed_5d_branch5x5_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_115 = torch.nn.functional.batch_norm(x_114, l__mod___mixed_5d_branch5x5_2_bn_running_mean, l__mod___mixed_5d_branch5x5_2_bn_running_var, l__mod___mixed_5d_branch5x5_2_bn_weight, l__mod___mixed_5d_branch5x5_2_bn_bias, True, 0.1, 0.001);  x_114 = l__mod___mixed_5d_branch5x5_2_bn_running_mean = l__mod___mixed_5d_branch5x5_2_bn_running_var = l__mod___mixed_5d_branch5x5_2_bn_weight = l__mod___mixed_5d_branch5x5_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_116 = self.L__mod___Mixed_5d_branch5x5_2_bn_drop(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch5x5_5 = self.L__mod___Mixed_5d_branch5x5_2_bn_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_119 = self.L__mod___Mixed_5d_branch3x3dbl_1_conv(x_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5d_branch3x3dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_5d_branch3x3dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__22 = l__mod___mixed_5d_branch3x3dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_5d_branch3x3dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch3x3dbl_1_bn_running_mean = self.L__mod___Mixed_5d_branch3x3dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch3x3dbl_1_bn_running_var = self.L__mod___Mixed_5d_branch3x3dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5d_branch3x3dbl_1_bn_weight = self.L__mod___Mixed_5d_branch3x3dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5d_branch3x3dbl_1_bn_bias = self.L__mod___Mixed_5d_branch3x3dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_120 = torch.nn.functional.batch_norm(x_119, l__mod___mixed_5d_branch3x3dbl_1_bn_running_mean, l__mod___mixed_5d_branch3x3dbl_1_bn_running_var, l__mod___mixed_5d_branch3x3dbl_1_bn_weight, l__mod___mixed_5d_branch3x3dbl_1_bn_bias, True, 0.1, 0.001);  x_119 = l__mod___mixed_5d_branch3x3dbl_1_bn_running_mean = l__mod___mixed_5d_branch3x3dbl_1_bn_running_var = l__mod___mixed_5d_branch3x3dbl_1_bn_weight = l__mod___mixed_5d_branch3x3dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_121 = self.L__mod___Mixed_5d_branch3x3dbl_1_bn_drop(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_6 = self.L__mod___Mixed_5d_branch3x3dbl_1_bn_act(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_124 = self.L__mod___Mixed_5d_branch3x3dbl_2_conv(branch3x3dbl_6);  branch3x3dbl_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5d_branch3x3dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_5d_branch3x3dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__23 = l__mod___mixed_5d_branch3x3dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_5d_branch3x3dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch3x3dbl_2_bn_running_mean = self.L__mod___Mixed_5d_branch3x3dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch3x3dbl_2_bn_running_var = self.L__mod___Mixed_5d_branch3x3dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5d_branch3x3dbl_2_bn_weight = self.L__mod___Mixed_5d_branch3x3dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5d_branch3x3dbl_2_bn_bias = self.L__mod___Mixed_5d_branch3x3dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_125 = torch.nn.functional.batch_norm(x_124, l__mod___mixed_5d_branch3x3dbl_2_bn_running_mean, l__mod___mixed_5d_branch3x3dbl_2_bn_running_var, l__mod___mixed_5d_branch3x3dbl_2_bn_weight, l__mod___mixed_5d_branch3x3dbl_2_bn_bias, True, 0.1, 0.001);  x_124 = l__mod___mixed_5d_branch3x3dbl_2_bn_running_mean = l__mod___mixed_5d_branch3x3dbl_2_bn_running_var = l__mod___mixed_5d_branch3x3dbl_2_bn_weight = l__mod___mixed_5d_branch3x3dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_126 = self.L__mod___Mixed_5d_branch3x3dbl_2_bn_drop(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_7 = self.L__mod___Mixed_5d_branch3x3dbl_2_bn_act(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_129 = self.L__mod___Mixed_5d_branch3x3dbl_3_conv(branch3x3dbl_7);  branch3x3dbl_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5d_branch3x3dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_5d_branch3x3dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__24 = l__mod___mixed_5d_branch3x3dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_5d_branch3x3dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch3x3dbl_3_bn_running_mean = self.L__mod___Mixed_5d_branch3x3dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch3x3dbl_3_bn_running_var = self.L__mod___Mixed_5d_branch3x3dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5d_branch3x3dbl_3_bn_weight = self.L__mod___Mixed_5d_branch3x3dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5d_branch3x3dbl_3_bn_bias = self.L__mod___Mixed_5d_branch3x3dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_130 = torch.nn.functional.batch_norm(x_129, l__mod___mixed_5d_branch3x3dbl_3_bn_running_mean, l__mod___mixed_5d_branch3x3dbl_3_bn_running_var, l__mod___mixed_5d_branch3x3dbl_3_bn_weight, l__mod___mixed_5d_branch3x3dbl_3_bn_bias, True, 0.1, 0.001);  x_129 = l__mod___mixed_5d_branch3x3dbl_3_bn_running_mean = l__mod___mixed_5d_branch3x3dbl_3_bn_running_var = l__mod___mixed_5d_branch3x3dbl_3_bn_weight = l__mod___mixed_5d_branch3x3dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_131 = self.L__mod___Mixed_5d_branch3x3dbl_3_bn_drop(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_8 = self.L__mod___Mixed_5d_branch3x3dbl_3_bn_act(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_4 = torch._C._nn.avg_pool2d(x_103, kernel_size = 3, stride = 1, padding = 1);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_134 = self.L__mod___Mixed_5d_branch_pool_conv(branch_pool_4);  branch_pool_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_5d_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_5d_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__25 = l__mod___mixed_5d_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_5d_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch_pool_bn_running_mean = self.L__mod___Mixed_5d_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_5d_branch_pool_bn_running_var = self.L__mod___Mixed_5d_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_5d_branch_pool_bn_weight = self.L__mod___Mixed_5d_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_5d_branch_pool_bn_bias = self.L__mod___Mixed_5d_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_135 = torch.nn.functional.batch_norm(x_134, l__mod___mixed_5d_branch_pool_bn_running_mean, l__mod___mixed_5d_branch_pool_bn_running_var, l__mod___mixed_5d_branch_pool_bn_weight, l__mod___mixed_5d_branch_pool_bn_bias, True, 0.1, 0.001);  x_134 = l__mod___mixed_5d_branch_pool_bn_running_mean = l__mod___mixed_5d_branch_pool_bn_running_var = l__mod___mixed_5d_branch_pool_bn_weight = l__mod___mixed_5d_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_136 = self.L__mod___Mixed_5d_branch_pool_bn_drop(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_5 = self.L__mod___Mixed_5d_branch_pool_bn_act(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    x_139 = torch.cat([branch1x1_2, branch5x5_5, branch3x3dbl_8, branch_pool_5], 1);  branch1x1_2 = branch5x5_5 = branch3x3dbl_8 = branch_pool_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_140 = self.L__mod___Mixed_6a_branch3x3_conv(x_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6a_branch3x3_bn_num_batches_tracked = self.L__mod___Mixed_6a_branch3x3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__26 = l__mod___mixed_6a_branch3x3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6a_branch3x3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3_bn_running_mean = self.L__mod___Mixed_6a_branch3x3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3_bn_running_var = self.L__mod___Mixed_6a_branch3x3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6a_branch3x3_bn_weight = self.L__mod___Mixed_6a_branch3x3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6a_branch3x3_bn_bias = self.L__mod___Mixed_6a_branch3x3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_141 = torch.nn.functional.batch_norm(x_140, l__mod___mixed_6a_branch3x3_bn_running_mean, l__mod___mixed_6a_branch3x3_bn_running_var, l__mod___mixed_6a_branch3x3_bn_weight, l__mod___mixed_6a_branch3x3_bn_bias, True, 0.1, 0.001);  x_140 = l__mod___mixed_6a_branch3x3_bn_running_mean = l__mod___mixed_6a_branch3x3_bn_running_var = l__mod___mixed_6a_branch3x3_bn_weight = l__mod___mixed_6a_branch3x3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_142 = self.L__mod___Mixed_6a_branch3x3_bn_drop(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3 = self.L__mod___Mixed_6a_branch3x3_bn_act(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_145 = self.L__mod___Mixed_6a_branch3x3dbl_1_conv(x_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6a_branch3x3dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_6a_branch3x3dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__27 = l__mod___mixed_6a_branch3x3dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6a_branch3x3dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3dbl_1_bn_running_mean = self.L__mod___Mixed_6a_branch3x3dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3dbl_1_bn_running_var = self.L__mod___Mixed_6a_branch3x3dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6a_branch3x3dbl_1_bn_weight = self.L__mod___Mixed_6a_branch3x3dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6a_branch3x3dbl_1_bn_bias = self.L__mod___Mixed_6a_branch3x3dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_146 = torch.nn.functional.batch_norm(x_145, l__mod___mixed_6a_branch3x3dbl_1_bn_running_mean, l__mod___mixed_6a_branch3x3dbl_1_bn_running_var, l__mod___mixed_6a_branch3x3dbl_1_bn_weight, l__mod___mixed_6a_branch3x3dbl_1_bn_bias, True, 0.1, 0.001);  x_145 = l__mod___mixed_6a_branch3x3dbl_1_bn_running_mean = l__mod___mixed_6a_branch3x3dbl_1_bn_running_var = l__mod___mixed_6a_branch3x3dbl_1_bn_weight = l__mod___mixed_6a_branch3x3dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_147 = self.L__mod___Mixed_6a_branch3x3dbl_1_bn_drop(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_9 = self.L__mod___Mixed_6a_branch3x3dbl_1_bn_act(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_150 = self.L__mod___Mixed_6a_branch3x3dbl_2_conv(branch3x3dbl_9);  branch3x3dbl_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6a_branch3x3dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_6a_branch3x3dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__28 = l__mod___mixed_6a_branch3x3dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6a_branch3x3dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3dbl_2_bn_running_mean = self.L__mod___Mixed_6a_branch3x3dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3dbl_2_bn_running_var = self.L__mod___Mixed_6a_branch3x3dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6a_branch3x3dbl_2_bn_weight = self.L__mod___Mixed_6a_branch3x3dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6a_branch3x3dbl_2_bn_bias = self.L__mod___Mixed_6a_branch3x3dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_151 = torch.nn.functional.batch_norm(x_150, l__mod___mixed_6a_branch3x3dbl_2_bn_running_mean, l__mod___mixed_6a_branch3x3dbl_2_bn_running_var, l__mod___mixed_6a_branch3x3dbl_2_bn_weight, l__mod___mixed_6a_branch3x3dbl_2_bn_bias, True, 0.1, 0.001);  x_150 = l__mod___mixed_6a_branch3x3dbl_2_bn_running_mean = l__mod___mixed_6a_branch3x3dbl_2_bn_running_var = l__mod___mixed_6a_branch3x3dbl_2_bn_weight = l__mod___mixed_6a_branch3x3dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_152 = self.L__mod___Mixed_6a_branch3x3dbl_2_bn_drop(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_10 = self.L__mod___Mixed_6a_branch3x3dbl_2_bn_act(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_155 = self.L__mod___Mixed_6a_branch3x3dbl_3_conv(branch3x3dbl_10);  branch3x3dbl_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6a_branch3x3dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_6a_branch3x3dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__29 = l__mod___mixed_6a_branch3x3dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6a_branch3x3dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3dbl_3_bn_running_mean = self.L__mod___Mixed_6a_branch3x3dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6a_branch3x3dbl_3_bn_running_var = self.L__mod___Mixed_6a_branch3x3dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6a_branch3x3dbl_3_bn_weight = self.L__mod___Mixed_6a_branch3x3dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6a_branch3x3dbl_3_bn_bias = self.L__mod___Mixed_6a_branch3x3dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_156 = torch.nn.functional.batch_norm(x_155, l__mod___mixed_6a_branch3x3dbl_3_bn_running_mean, l__mod___mixed_6a_branch3x3dbl_3_bn_running_var, l__mod___mixed_6a_branch3x3dbl_3_bn_weight, l__mod___mixed_6a_branch3x3dbl_3_bn_bias, True, 0.1, 0.001);  x_155 = l__mod___mixed_6a_branch3x3dbl_3_bn_running_mean = l__mod___mixed_6a_branch3x3dbl_3_bn_running_var = l__mod___mixed_6a_branch3x3dbl_3_bn_weight = l__mod___mixed_6a_branch3x3dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_157 = self.L__mod___Mixed_6a_branch3x3dbl_3_bn_drop(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_11 = self.L__mod___Mixed_6a_branch3x3dbl_3_bn_act(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:77, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
    branch_pool_6 = torch.nn.functional.max_pool2d(x_139, kernel_size = 3, stride = 2);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:84, code: return torch.cat(outputs, 1)
    x_160 = torch.cat([branch3x3, branch3x3dbl_11, branch_pool_6], 1);  branch3x3 = branch3x3dbl_11 = branch_pool_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_161 = self.L__mod___Mixed_6b_branch1x1_conv(x_160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__30 = l__mod___mixed_6b_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch1x1_bn_running_mean = self.L__mod___Mixed_6b_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch1x1_bn_running_var = self.L__mod___Mixed_6b_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch1x1_bn_weight = self.L__mod___Mixed_6b_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch1x1_bn_bias = self.L__mod___Mixed_6b_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_162 = torch.nn.functional.batch_norm(x_161, l__mod___mixed_6b_branch1x1_bn_running_mean, l__mod___mixed_6b_branch1x1_bn_running_var, l__mod___mixed_6b_branch1x1_bn_weight, l__mod___mixed_6b_branch1x1_bn_bias, True, 0.1, 0.001);  x_161 = l__mod___mixed_6b_branch1x1_bn_running_mean = l__mod___mixed_6b_branch1x1_bn_running_var = l__mod___mixed_6b_branch1x1_bn_weight = l__mod___mixed_6b_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_163 = self.L__mod___Mixed_6b_branch1x1_bn_drop(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_3 = self.L__mod___Mixed_6b_branch1x1_bn_act(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_166 = self.L__mod___Mixed_6b_branch7x7_1_conv(x_160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7_1_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__31 = l__mod___mixed_6b_branch7x7_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7_1_bn_running_mean = self.L__mod___Mixed_6b_branch7x7_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7_1_bn_running_var = self.L__mod___Mixed_6b_branch7x7_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7_1_bn_weight = self.L__mod___Mixed_6b_branch7x7_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7_1_bn_bias = self.L__mod___Mixed_6b_branch7x7_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_167 = torch.nn.functional.batch_norm(x_166, l__mod___mixed_6b_branch7x7_1_bn_running_mean, l__mod___mixed_6b_branch7x7_1_bn_running_var, l__mod___mixed_6b_branch7x7_1_bn_weight, l__mod___mixed_6b_branch7x7_1_bn_bias, True, 0.1, 0.001);  x_166 = l__mod___mixed_6b_branch7x7_1_bn_running_mean = l__mod___mixed_6b_branch7x7_1_bn_running_var = l__mod___mixed_6b_branch7x7_1_bn_weight = l__mod___mixed_6b_branch7x7_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_168 = self.L__mod___Mixed_6b_branch7x7_1_bn_drop(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7 = self.L__mod___Mixed_6b_branch7x7_1_bn_act(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_171 = self.L__mod___Mixed_6b_branch7x7_2_conv(branch7x7);  branch7x7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7_2_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__32 = l__mod___mixed_6b_branch7x7_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7_2_bn_running_mean = self.L__mod___Mixed_6b_branch7x7_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7_2_bn_running_var = self.L__mod___Mixed_6b_branch7x7_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7_2_bn_weight = self.L__mod___Mixed_6b_branch7x7_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7_2_bn_bias = self.L__mod___Mixed_6b_branch7x7_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_172 = torch.nn.functional.batch_norm(x_171, l__mod___mixed_6b_branch7x7_2_bn_running_mean, l__mod___mixed_6b_branch7x7_2_bn_running_var, l__mod___mixed_6b_branch7x7_2_bn_weight, l__mod___mixed_6b_branch7x7_2_bn_bias, True, 0.1, 0.001);  x_171 = l__mod___mixed_6b_branch7x7_2_bn_running_mean = l__mod___mixed_6b_branch7x7_2_bn_running_var = l__mod___mixed_6b_branch7x7_2_bn_weight = l__mod___mixed_6b_branch7x7_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_173 = self.L__mod___Mixed_6b_branch7x7_2_bn_drop(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_1 = self.L__mod___Mixed_6b_branch7x7_2_bn_act(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_176 = self.L__mod___Mixed_6b_branch7x7_3_conv(branch7x7_1);  branch7x7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7_3_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__33 = l__mod___mixed_6b_branch7x7_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7_3_bn_running_mean = self.L__mod___Mixed_6b_branch7x7_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7_3_bn_running_var = self.L__mod___Mixed_6b_branch7x7_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7_3_bn_weight = self.L__mod___Mixed_6b_branch7x7_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7_3_bn_bias = self.L__mod___Mixed_6b_branch7x7_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_177 = torch.nn.functional.batch_norm(x_176, l__mod___mixed_6b_branch7x7_3_bn_running_mean, l__mod___mixed_6b_branch7x7_3_bn_running_var, l__mod___mixed_6b_branch7x7_3_bn_weight, l__mod___mixed_6b_branch7x7_3_bn_bias, True, 0.1, 0.001);  x_176 = l__mod___mixed_6b_branch7x7_3_bn_running_mean = l__mod___mixed_6b_branch7x7_3_bn_running_var = l__mod___mixed_6b_branch7x7_3_bn_weight = l__mod___mixed_6b_branch7x7_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_178 = self.L__mod___Mixed_6b_branch7x7_3_bn_drop(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_2 = self.L__mod___Mixed_6b_branch7x7_3_bn_act(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_181 = self.L__mod___Mixed_6b_branch7x7dbl_1_conv(x_160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__34 = l__mod___mixed_6b_branch7x7dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_1_bn_running_mean = self.L__mod___Mixed_6b_branch7x7dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_1_bn_running_var = self.L__mod___Mixed_6b_branch7x7dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7dbl_1_bn_weight = self.L__mod___Mixed_6b_branch7x7dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7dbl_1_bn_bias = self.L__mod___Mixed_6b_branch7x7dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_182 = torch.nn.functional.batch_norm(x_181, l__mod___mixed_6b_branch7x7dbl_1_bn_running_mean, l__mod___mixed_6b_branch7x7dbl_1_bn_running_var, l__mod___mixed_6b_branch7x7dbl_1_bn_weight, l__mod___mixed_6b_branch7x7dbl_1_bn_bias, True, 0.1, 0.001);  x_181 = l__mod___mixed_6b_branch7x7dbl_1_bn_running_mean = l__mod___mixed_6b_branch7x7dbl_1_bn_running_var = l__mod___mixed_6b_branch7x7dbl_1_bn_weight = l__mod___mixed_6b_branch7x7dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_183 = self.L__mod___Mixed_6b_branch7x7dbl_1_bn_drop(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl = self.L__mod___Mixed_6b_branch7x7dbl_1_bn_act(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_186 = self.L__mod___Mixed_6b_branch7x7dbl_2_conv(branch7x7dbl);  branch7x7dbl = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__35 = l__mod___mixed_6b_branch7x7dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_2_bn_running_mean = self.L__mod___Mixed_6b_branch7x7dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_2_bn_running_var = self.L__mod___Mixed_6b_branch7x7dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7dbl_2_bn_weight = self.L__mod___Mixed_6b_branch7x7dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7dbl_2_bn_bias = self.L__mod___Mixed_6b_branch7x7dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_187 = torch.nn.functional.batch_norm(x_186, l__mod___mixed_6b_branch7x7dbl_2_bn_running_mean, l__mod___mixed_6b_branch7x7dbl_2_bn_running_var, l__mod___mixed_6b_branch7x7dbl_2_bn_weight, l__mod___mixed_6b_branch7x7dbl_2_bn_bias, True, 0.1, 0.001);  x_186 = l__mod___mixed_6b_branch7x7dbl_2_bn_running_mean = l__mod___mixed_6b_branch7x7dbl_2_bn_running_var = l__mod___mixed_6b_branch7x7dbl_2_bn_weight = l__mod___mixed_6b_branch7x7dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_188 = self.L__mod___Mixed_6b_branch7x7dbl_2_bn_drop(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_1 = self.L__mod___Mixed_6b_branch7x7dbl_2_bn_act(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_191 = self.L__mod___Mixed_6b_branch7x7dbl_3_conv(branch7x7dbl_1);  branch7x7dbl_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__36 = l__mod___mixed_6b_branch7x7dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_3_bn_running_mean = self.L__mod___Mixed_6b_branch7x7dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_3_bn_running_var = self.L__mod___Mixed_6b_branch7x7dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7dbl_3_bn_weight = self.L__mod___Mixed_6b_branch7x7dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7dbl_3_bn_bias = self.L__mod___Mixed_6b_branch7x7dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_192 = torch.nn.functional.batch_norm(x_191, l__mod___mixed_6b_branch7x7dbl_3_bn_running_mean, l__mod___mixed_6b_branch7x7dbl_3_bn_running_var, l__mod___mixed_6b_branch7x7dbl_3_bn_weight, l__mod___mixed_6b_branch7x7dbl_3_bn_bias, True, 0.1, 0.001);  x_191 = l__mod___mixed_6b_branch7x7dbl_3_bn_running_mean = l__mod___mixed_6b_branch7x7dbl_3_bn_running_var = l__mod___mixed_6b_branch7x7dbl_3_bn_weight = l__mod___mixed_6b_branch7x7dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_193 = self.L__mod___Mixed_6b_branch7x7dbl_3_bn_drop(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_2 = self.L__mod___Mixed_6b_branch7x7dbl_3_bn_act(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_196 = self.L__mod___Mixed_6b_branch7x7dbl_4_conv(branch7x7dbl_2);  branch7x7dbl_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7dbl_4_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7dbl_4_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__37 = l__mod___mixed_6b_branch7x7dbl_4_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7dbl_4_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_4_bn_running_mean = self.L__mod___Mixed_6b_branch7x7dbl_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_4_bn_running_var = self.L__mod___Mixed_6b_branch7x7dbl_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7dbl_4_bn_weight = self.L__mod___Mixed_6b_branch7x7dbl_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7dbl_4_bn_bias = self.L__mod___Mixed_6b_branch7x7dbl_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_197 = torch.nn.functional.batch_norm(x_196, l__mod___mixed_6b_branch7x7dbl_4_bn_running_mean, l__mod___mixed_6b_branch7x7dbl_4_bn_running_var, l__mod___mixed_6b_branch7x7dbl_4_bn_weight, l__mod___mixed_6b_branch7x7dbl_4_bn_bias, True, 0.1, 0.001);  x_196 = l__mod___mixed_6b_branch7x7dbl_4_bn_running_mean = l__mod___mixed_6b_branch7x7dbl_4_bn_running_var = l__mod___mixed_6b_branch7x7dbl_4_bn_weight = l__mod___mixed_6b_branch7x7dbl_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_198 = self.L__mod___Mixed_6b_branch7x7dbl_4_bn_drop(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_3 = self.L__mod___Mixed_6b_branch7x7dbl_4_bn_act(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_201 = self.L__mod___Mixed_6b_branch7x7dbl_5_conv(branch7x7dbl_3);  branch7x7dbl_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch7x7dbl_5_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch7x7dbl_5_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__38 = l__mod___mixed_6b_branch7x7dbl_5_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch7x7dbl_5_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_5_bn_running_mean = self.L__mod___Mixed_6b_branch7x7dbl_5_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch7x7dbl_5_bn_running_var = self.L__mod___Mixed_6b_branch7x7dbl_5_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch7x7dbl_5_bn_weight = self.L__mod___Mixed_6b_branch7x7dbl_5_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch7x7dbl_5_bn_bias = self.L__mod___Mixed_6b_branch7x7dbl_5_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_202 = torch.nn.functional.batch_norm(x_201, l__mod___mixed_6b_branch7x7dbl_5_bn_running_mean, l__mod___mixed_6b_branch7x7dbl_5_bn_running_var, l__mod___mixed_6b_branch7x7dbl_5_bn_weight, l__mod___mixed_6b_branch7x7dbl_5_bn_bias, True, 0.1, 0.001);  x_201 = l__mod___mixed_6b_branch7x7dbl_5_bn_running_mean = l__mod___mixed_6b_branch7x7dbl_5_bn_running_var = l__mod___mixed_6b_branch7x7dbl_5_bn_weight = l__mod___mixed_6b_branch7x7dbl_5_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_203 = self.L__mod___Mixed_6b_branch7x7dbl_5_bn_drop(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_4 = self.L__mod___Mixed_6b_branch7x7dbl_5_bn_act(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_7 = torch._C._nn.avg_pool2d(x_160, kernel_size = 3, stride = 1, padding = 1);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_206 = self.L__mod___Mixed_6b_branch_pool_conv(branch_pool_7);  branch_pool_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6b_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_6b_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__39 = l__mod___mixed_6b_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_6b_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch_pool_bn_running_mean = self.L__mod___Mixed_6b_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6b_branch_pool_bn_running_var = self.L__mod___Mixed_6b_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6b_branch_pool_bn_weight = self.L__mod___Mixed_6b_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6b_branch_pool_bn_bias = self.L__mod___Mixed_6b_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_207 = torch.nn.functional.batch_norm(x_206, l__mod___mixed_6b_branch_pool_bn_running_mean, l__mod___mixed_6b_branch_pool_bn_running_var, l__mod___mixed_6b_branch_pool_bn_weight, l__mod___mixed_6b_branch_pool_bn_bias, True, 0.1, 0.001);  x_206 = l__mod___mixed_6b_branch_pool_bn_running_mean = l__mod___mixed_6b_branch_pool_bn_running_var = l__mod___mixed_6b_branch_pool_bn_weight = l__mod___mixed_6b_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_208 = self.L__mod___Mixed_6b_branch_pool_bn_drop(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_8 = self.L__mod___Mixed_6b_branch_pool_bn_act(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    x_211 = torch.cat([branch1x1_3, branch7x7_2, branch7x7dbl_4, branch_pool_8], 1);  branch1x1_3 = branch7x7_2 = branch7x7dbl_4 = branch_pool_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_212 = self.L__mod___Mixed_6c_branch1x1_conv(x_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__40 = l__mod___mixed_6c_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch1x1_bn_running_mean = self.L__mod___Mixed_6c_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch1x1_bn_running_var = self.L__mod___Mixed_6c_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch1x1_bn_weight = self.L__mod___Mixed_6c_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch1x1_bn_bias = self.L__mod___Mixed_6c_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_213 = torch.nn.functional.batch_norm(x_212, l__mod___mixed_6c_branch1x1_bn_running_mean, l__mod___mixed_6c_branch1x1_bn_running_var, l__mod___mixed_6c_branch1x1_bn_weight, l__mod___mixed_6c_branch1x1_bn_bias, True, 0.1, 0.001);  x_212 = l__mod___mixed_6c_branch1x1_bn_running_mean = l__mod___mixed_6c_branch1x1_bn_running_var = l__mod___mixed_6c_branch1x1_bn_weight = l__mod___mixed_6c_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_214 = self.L__mod___Mixed_6c_branch1x1_bn_drop(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_4 = self.L__mod___Mixed_6c_branch1x1_bn_act(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_217 = self.L__mod___Mixed_6c_branch7x7_1_conv(x_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7_1_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__41 = l__mod___mixed_6c_branch7x7_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7_1_bn_running_mean = self.L__mod___Mixed_6c_branch7x7_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7_1_bn_running_var = self.L__mod___Mixed_6c_branch7x7_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7_1_bn_weight = self.L__mod___Mixed_6c_branch7x7_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7_1_bn_bias = self.L__mod___Mixed_6c_branch7x7_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_218 = torch.nn.functional.batch_norm(x_217, l__mod___mixed_6c_branch7x7_1_bn_running_mean, l__mod___mixed_6c_branch7x7_1_bn_running_var, l__mod___mixed_6c_branch7x7_1_bn_weight, l__mod___mixed_6c_branch7x7_1_bn_bias, True, 0.1, 0.001);  x_217 = l__mod___mixed_6c_branch7x7_1_bn_running_mean = l__mod___mixed_6c_branch7x7_1_bn_running_var = l__mod___mixed_6c_branch7x7_1_bn_weight = l__mod___mixed_6c_branch7x7_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_219 = self.L__mod___Mixed_6c_branch7x7_1_bn_drop(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_3 = self.L__mod___Mixed_6c_branch7x7_1_bn_act(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_222 = self.L__mod___Mixed_6c_branch7x7_2_conv(branch7x7_3);  branch7x7_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7_2_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__42 = l__mod___mixed_6c_branch7x7_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7_2_bn_running_mean = self.L__mod___Mixed_6c_branch7x7_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7_2_bn_running_var = self.L__mod___Mixed_6c_branch7x7_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7_2_bn_weight = self.L__mod___Mixed_6c_branch7x7_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7_2_bn_bias = self.L__mod___Mixed_6c_branch7x7_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_223 = torch.nn.functional.batch_norm(x_222, l__mod___mixed_6c_branch7x7_2_bn_running_mean, l__mod___mixed_6c_branch7x7_2_bn_running_var, l__mod___mixed_6c_branch7x7_2_bn_weight, l__mod___mixed_6c_branch7x7_2_bn_bias, True, 0.1, 0.001);  x_222 = l__mod___mixed_6c_branch7x7_2_bn_running_mean = l__mod___mixed_6c_branch7x7_2_bn_running_var = l__mod___mixed_6c_branch7x7_2_bn_weight = l__mod___mixed_6c_branch7x7_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_224 = self.L__mod___Mixed_6c_branch7x7_2_bn_drop(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_4 = self.L__mod___Mixed_6c_branch7x7_2_bn_act(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_227 = self.L__mod___Mixed_6c_branch7x7_3_conv(branch7x7_4);  branch7x7_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7_3_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__43 = l__mod___mixed_6c_branch7x7_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7_3_bn_running_mean = self.L__mod___Mixed_6c_branch7x7_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7_3_bn_running_var = self.L__mod___Mixed_6c_branch7x7_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7_3_bn_weight = self.L__mod___Mixed_6c_branch7x7_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7_3_bn_bias = self.L__mod___Mixed_6c_branch7x7_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, l__mod___mixed_6c_branch7x7_3_bn_running_mean, l__mod___mixed_6c_branch7x7_3_bn_running_var, l__mod___mixed_6c_branch7x7_3_bn_weight, l__mod___mixed_6c_branch7x7_3_bn_bias, True, 0.1, 0.001);  x_227 = l__mod___mixed_6c_branch7x7_3_bn_running_mean = l__mod___mixed_6c_branch7x7_3_bn_running_var = l__mod___mixed_6c_branch7x7_3_bn_weight = l__mod___mixed_6c_branch7x7_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.L__mod___Mixed_6c_branch7x7_3_bn_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_5 = self.L__mod___Mixed_6c_branch7x7_3_bn_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_232 = self.L__mod___Mixed_6c_branch7x7dbl_1_conv(x_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__44 = l__mod___mixed_6c_branch7x7dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_1_bn_running_mean = self.L__mod___Mixed_6c_branch7x7dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_1_bn_running_var = self.L__mod___Mixed_6c_branch7x7dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7dbl_1_bn_weight = self.L__mod___Mixed_6c_branch7x7dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7dbl_1_bn_bias = self.L__mod___Mixed_6c_branch7x7dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_233 = torch.nn.functional.batch_norm(x_232, l__mod___mixed_6c_branch7x7dbl_1_bn_running_mean, l__mod___mixed_6c_branch7x7dbl_1_bn_running_var, l__mod___mixed_6c_branch7x7dbl_1_bn_weight, l__mod___mixed_6c_branch7x7dbl_1_bn_bias, True, 0.1, 0.001);  x_232 = l__mod___mixed_6c_branch7x7dbl_1_bn_running_mean = l__mod___mixed_6c_branch7x7dbl_1_bn_running_var = l__mod___mixed_6c_branch7x7dbl_1_bn_weight = l__mod___mixed_6c_branch7x7dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_234 = self.L__mod___Mixed_6c_branch7x7dbl_1_bn_drop(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_5 = self.L__mod___Mixed_6c_branch7x7dbl_1_bn_act(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_237 = self.L__mod___Mixed_6c_branch7x7dbl_2_conv(branch7x7dbl_5);  branch7x7dbl_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__45 = l__mod___mixed_6c_branch7x7dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_2_bn_running_mean = self.L__mod___Mixed_6c_branch7x7dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_2_bn_running_var = self.L__mod___Mixed_6c_branch7x7dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7dbl_2_bn_weight = self.L__mod___Mixed_6c_branch7x7dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7dbl_2_bn_bias = self.L__mod___Mixed_6c_branch7x7dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_238 = torch.nn.functional.batch_norm(x_237, l__mod___mixed_6c_branch7x7dbl_2_bn_running_mean, l__mod___mixed_6c_branch7x7dbl_2_bn_running_var, l__mod___mixed_6c_branch7x7dbl_2_bn_weight, l__mod___mixed_6c_branch7x7dbl_2_bn_bias, True, 0.1, 0.001);  x_237 = l__mod___mixed_6c_branch7x7dbl_2_bn_running_mean = l__mod___mixed_6c_branch7x7dbl_2_bn_running_var = l__mod___mixed_6c_branch7x7dbl_2_bn_weight = l__mod___mixed_6c_branch7x7dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_239 = self.L__mod___Mixed_6c_branch7x7dbl_2_bn_drop(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_6 = self.L__mod___Mixed_6c_branch7x7dbl_2_bn_act(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_242 = self.L__mod___Mixed_6c_branch7x7dbl_3_conv(branch7x7dbl_6);  branch7x7dbl_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__46 = l__mod___mixed_6c_branch7x7dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_3_bn_running_mean = self.L__mod___Mixed_6c_branch7x7dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_3_bn_running_var = self.L__mod___Mixed_6c_branch7x7dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7dbl_3_bn_weight = self.L__mod___Mixed_6c_branch7x7dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7dbl_3_bn_bias = self.L__mod___Mixed_6c_branch7x7dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_243 = torch.nn.functional.batch_norm(x_242, l__mod___mixed_6c_branch7x7dbl_3_bn_running_mean, l__mod___mixed_6c_branch7x7dbl_3_bn_running_var, l__mod___mixed_6c_branch7x7dbl_3_bn_weight, l__mod___mixed_6c_branch7x7dbl_3_bn_bias, True, 0.1, 0.001);  x_242 = l__mod___mixed_6c_branch7x7dbl_3_bn_running_mean = l__mod___mixed_6c_branch7x7dbl_3_bn_running_var = l__mod___mixed_6c_branch7x7dbl_3_bn_weight = l__mod___mixed_6c_branch7x7dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_244 = self.L__mod___Mixed_6c_branch7x7dbl_3_bn_drop(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_7 = self.L__mod___Mixed_6c_branch7x7dbl_3_bn_act(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_247 = self.L__mod___Mixed_6c_branch7x7dbl_4_conv(branch7x7dbl_7);  branch7x7dbl_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7dbl_4_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7dbl_4_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__47 = l__mod___mixed_6c_branch7x7dbl_4_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7dbl_4_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_4_bn_running_mean = self.L__mod___Mixed_6c_branch7x7dbl_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_4_bn_running_var = self.L__mod___Mixed_6c_branch7x7dbl_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7dbl_4_bn_weight = self.L__mod___Mixed_6c_branch7x7dbl_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7dbl_4_bn_bias = self.L__mod___Mixed_6c_branch7x7dbl_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_248 = torch.nn.functional.batch_norm(x_247, l__mod___mixed_6c_branch7x7dbl_4_bn_running_mean, l__mod___mixed_6c_branch7x7dbl_4_bn_running_var, l__mod___mixed_6c_branch7x7dbl_4_bn_weight, l__mod___mixed_6c_branch7x7dbl_4_bn_bias, True, 0.1, 0.001);  x_247 = l__mod___mixed_6c_branch7x7dbl_4_bn_running_mean = l__mod___mixed_6c_branch7x7dbl_4_bn_running_var = l__mod___mixed_6c_branch7x7dbl_4_bn_weight = l__mod___mixed_6c_branch7x7dbl_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_249 = self.L__mod___Mixed_6c_branch7x7dbl_4_bn_drop(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_8 = self.L__mod___Mixed_6c_branch7x7dbl_4_bn_act(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_252 = self.L__mod___Mixed_6c_branch7x7dbl_5_conv(branch7x7dbl_8);  branch7x7dbl_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch7x7dbl_5_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch7x7dbl_5_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__48 = l__mod___mixed_6c_branch7x7dbl_5_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch7x7dbl_5_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_5_bn_running_mean = self.L__mod___Mixed_6c_branch7x7dbl_5_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch7x7dbl_5_bn_running_var = self.L__mod___Mixed_6c_branch7x7dbl_5_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch7x7dbl_5_bn_weight = self.L__mod___Mixed_6c_branch7x7dbl_5_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch7x7dbl_5_bn_bias = self.L__mod___Mixed_6c_branch7x7dbl_5_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_253 = torch.nn.functional.batch_norm(x_252, l__mod___mixed_6c_branch7x7dbl_5_bn_running_mean, l__mod___mixed_6c_branch7x7dbl_5_bn_running_var, l__mod___mixed_6c_branch7x7dbl_5_bn_weight, l__mod___mixed_6c_branch7x7dbl_5_bn_bias, True, 0.1, 0.001);  x_252 = l__mod___mixed_6c_branch7x7dbl_5_bn_running_mean = l__mod___mixed_6c_branch7x7dbl_5_bn_running_var = l__mod___mixed_6c_branch7x7dbl_5_bn_weight = l__mod___mixed_6c_branch7x7dbl_5_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_254 = self.L__mod___Mixed_6c_branch7x7dbl_5_bn_drop(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_9 = self.L__mod___Mixed_6c_branch7x7dbl_5_bn_act(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_9 = torch._C._nn.avg_pool2d(x_211, kernel_size = 3, stride = 1, padding = 1);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_257 = self.L__mod___Mixed_6c_branch_pool_conv(branch_pool_9);  branch_pool_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6c_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_6c_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__49 = l__mod___mixed_6c_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_6c_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch_pool_bn_running_mean = self.L__mod___Mixed_6c_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6c_branch_pool_bn_running_var = self.L__mod___Mixed_6c_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6c_branch_pool_bn_weight = self.L__mod___Mixed_6c_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6c_branch_pool_bn_bias = self.L__mod___Mixed_6c_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_258 = torch.nn.functional.batch_norm(x_257, l__mod___mixed_6c_branch_pool_bn_running_mean, l__mod___mixed_6c_branch_pool_bn_running_var, l__mod___mixed_6c_branch_pool_bn_weight, l__mod___mixed_6c_branch_pool_bn_bias, True, 0.1, 0.001);  x_257 = l__mod___mixed_6c_branch_pool_bn_running_mean = l__mod___mixed_6c_branch_pool_bn_running_var = l__mod___mixed_6c_branch_pool_bn_weight = l__mod___mixed_6c_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_259 = self.L__mod___Mixed_6c_branch_pool_bn_drop(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_10 = self.L__mod___Mixed_6c_branch_pool_bn_act(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    x_262 = torch.cat([branch1x1_4, branch7x7_5, branch7x7dbl_9, branch_pool_10], 1);  branch1x1_4 = branch7x7_5 = branch7x7dbl_9 = branch_pool_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_263 = self.L__mod___Mixed_6d_branch1x1_conv(x_262)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__50 = l__mod___mixed_6d_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch1x1_bn_running_mean = self.L__mod___Mixed_6d_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch1x1_bn_running_var = self.L__mod___Mixed_6d_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch1x1_bn_weight = self.L__mod___Mixed_6d_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch1x1_bn_bias = self.L__mod___Mixed_6d_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_264 = torch.nn.functional.batch_norm(x_263, l__mod___mixed_6d_branch1x1_bn_running_mean, l__mod___mixed_6d_branch1x1_bn_running_var, l__mod___mixed_6d_branch1x1_bn_weight, l__mod___mixed_6d_branch1x1_bn_bias, True, 0.1, 0.001);  x_263 = l__mod___mixed_6d_branch1x1_bn_running_mean = l__mod___mixed_6d_branch1x1_bn_running_var = l__mod___mixed_6d_branch1x1_bn_weight = l__mod___mixed_6d_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_265 = self.L__mod___Mixed_6d_branch1x1_bn_drop(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_5 = self.L__mod___Mixed_6d_branch1x1_bn_act(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_268 = self.L__mod___Mixed_6d_branch7x7_1_conv(x_262)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7_1_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__51 = l__mod___mixed_6d_branch7x7_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7_1_bn_running_mean = self.L__mod___Mixed_6d_branch7x7_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7_1_bn_running_var = self.L__mod___Mixed_6d_branch7x7_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7_1_bn_weight = self.L__mod___Mixed_6d_branch7x7_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7_1_bn_bias = self.L__mod___Mixed_6d_branch7x7_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_269 = torch.nn.functional.batch_norm(x_268, l__mod___mixed_6d_branch7x7_1_bn_running_mean, l__mod___mixed_6d_branch7x7_1_bn_running_var, l__mod___mixed_6d_branch7x7_1_bn_weight, l__mod___mixed_6d_branch7x7_1_bn_bias, True, 0.1, 0.001);  x_268 = l__mod___mixed_6d_branch7x7_1_bn_running_mean = l__mod___mixed_6d_branch7x7_1_bn_running_var = l__mod___mixed_6d_branch7x7_1_bn_weight = l__mod___mixed_6d_branch7x7_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_270 = self.L__mod___Mixed_6d_branch7x7_1_bn_drop(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_6 = self.L__mod___Mixed_6d_branch7x7_1_bn_act(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_273 = self.L__mod___Mixed_6d_branch7x7_2_conv(branch7x7_6);  branch7x7_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7_2_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__52 = l__mod___mixed_6d_branch7x7_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7_2_bn_running_mean = self.L__mod___Mixed_6d_branch7x7_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7_2_bn_running_var = self.L__mod___Mixed_6d_branch7x7_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7_2_bn_weight = self.L__mod___Mixed_6d_branch7x7_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7_2_bn_bias = self.L__mod___Mixed_6d_branch7x7_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_274 = torch.nn.functional.batch_norm(x_273, l__mod___mixed_6d_branch7x7_2_bn_running_mean, l__mod___mixed_6d_branch7x7_2_bn_running_var, l__mod___mixed_6d_branch7x7_2_bn_weight, l__mod___mixed_6d_branch7x7_2_bn_bias, True, 0.1, 0.001);  x_273 = l__mod___mixed_6d_branch7x7_2_bn_running_mean = l__mod___mixed_6d_branch7x7_2_bn_running_var = l__mod___mixed_6d_branch7x7_2_bn_weight = l__mod___mixed_6d_branch7x7_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_275 = self.L__mod___Mixed_6d_branch7x7_2_bn_drop(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_7 = self.L__mod___Mixed_6d_branch7x7_2_bn_act(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_278 = self.L__mod___Mixed_6d_branch7x7_3_conv(branch7x7_7);  branch7x7_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7_3_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__53 = l__mod___mixed_6d_branch7x7_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7_3_bn_running_mean = self.L__mod___Mixed_6d_branch7x7_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7_3_bn_running_var = self.L__mod___Mixed_6d_branch7x7_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7_3_bn_weight = self.L__mod___Mixed_6d_branch7x7_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7_3_bn_bias = self.L__mod___Mixed_6d_branch7x7_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_279 = torch.nn.functional.batch_norm(x_278, l__mod___mixed_6d_branch7x7_3_bn_running_mean, l__mod___mixed_6d_branch7x7_3_bn_running_var, l__mod___mixed_6d_branch7x7_3_bn_weight, l__mod___mixed_6d_branch7x7_3_bn_bias, True, 0.1, 0.001);  x_278 = l__mod___mixed_6d_branch7x7_3_bn_running_mean = l__mod___mixed_6d_branch7x7_3_bn_running_var = l__mod___mixed_6d_branch7x7_3_bn_weight = l__mod___mixed_6d_branch7x7_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_280 = self.L__mod___Mixed_6d_branch7x7_3_bn_drop(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_8 = self.L__mod___Mixed_6d_branch7x7_3_bn_act(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_283 = self.L__mod___Mixed_6d_branch7x7dbl_1_conv(x_262)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__54 = l__mod___mixed_6d_branch7x7dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_1_bn_running_mean = self.L__mod___Mixed_6d_branch7x7dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_1_bn_running_var = self.L__mod___Mixed_6d_branch7x7dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7dbl_1_bn_weight = self.L__mod___Mixed_6d_branch7x7dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7dbl_1_bn_bias = self.L__mod___Mixed_6d_branch7x7dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_284 = torch.nn.functional.batch_norm(x_283, l__mod___mixed_6d_branch7x7dbl_1_bn_running_mean, l__mod___mixed_6d_branch7x7dbl_1_bn_running_var, l__mod___mixed_6d_branch7x7dbl_1_bn_weight, l__mod___mixed_6d_branch7x7dbl_1_bn_bias, True, 0.1, 0.001);  x_283 = l__mod___mixed_6d_branch7x7dbl_1_bn_running_mean = l__mod___mixed_6d_branch7x7dbl_1_bn_running_var = l__mod___mixed_6d_branch7x7dbl_1_bn_weight = l__mod___mixed_6d_branch7x7dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_285 = self.L__mod___Mixed_6d_branch7x7dbl_1_bn_drop(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_10 = self.L__mod___Mixed_6d_branch7x7dbl_1_bn_act(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_288 = self.L__mod___Mixed_6d_branch7x7dbl_2_conv(branch7x7dbl_10);  branch7x7dbl_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__55 = l__mod___mixed_6d_branch7x7dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_2_bn_running_mean = self.L__mod___Mixed_6d_branch7x7dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_2_bn_running_var = self.L__mod___Mixed_6d_branch7x7dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7dbl_2_bn_weight = self.L__mod___Mixed_6d_branch7x7dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7dbl_2_bn_bias = self.L__mod___Mixed_6d_branch7x7dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_289 = torch.nn.functional.batch_norm(x_288, l__mod___mixed_6d_branch7x7dbl_2_bn_running_mean, l__mod___mixed_6d_branch7x7dbl_2_bn_running_var, l__mod___mixed_6d_branch7x7dbl_2_bn_weight, l__mod___mixed_6d_branch7x7dbl_2_bn_bias, True, 0.1, 0.001);  x_288 = l__mod___mixed_6d_branch7x7dbl_2_bn_running_mean = l__mod___mixed_6d_branch7x7dbl_2_bn_running_var = l__mod___mixed_6d_branch7x7dbl_2_bn_weight = l__mod___mixed_6d_branch7x7dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_290 = self.L__mod___Mixed_6d_branch7x7dbl_2_bn_drop(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_11 = self.L__mod___Mixed_6d_branch7x7dbl_2_bn_act(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_293 = self.L__mod___Mixed_6d_branch7x7dbl_3_conv(branch7x7dbl_11);  branch7x7dbl_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__56 = l__mod___mixed_6d_branch7x7dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_3_bn_running_mean = self.L__mod___Mixed_6d_branch7x7dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_3_bn_running_var = self.L__mod___Mixed_6d_branch7x7dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7dbl_3_bn_weight = self.L__mod___Mixed_6d_branch7x7dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7dbl_3_bn_bias = self.L__mod___Mixed_6d_branch7x7dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_294 = torch.nn.functional.batch_norm(x_293, l__mod___mixed_6d_branch7x7dbl_3_bn_running_mean, l__mod___mixed_6d_branch7x7dbl_3_bn_running_var, l__mod___mixed_6d_branch7x7dbl_3_bn_weight, l__mod___mixed_6d_branch7x7dbl_3_bn_bias, True, 0.1, 0.001);  x_293 = l__mod___mixed_6d_branch7x7dbl_3_bn_running_mean = l__mod___mixed_6d_branch7x7dbl_3_bn_running_var = l__mod___mixed_6d_branch7x7dbl_3_bn_weight = l__mod___mixed_6d_branch7x7dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_295 = self.L__mod___Mixed_6d_branch7x7dbl_3_bn_drop(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_12 = self.L__mod___Mixed_6d_branch7x7dbl_3_bn_act(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_298 = self.L__mod___Mixed_6d_branch7x7dbl_4_conv(branch7x7dbl_12);  branch7x7dbl_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7dbl_4_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7dbl_4_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__57 = l__mod___mixed_6d_branch7x7dbl_4_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7dbl_4_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_4_bn_running_mean = self.L__mod___Mixed_6d_branch7x7dbl_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_4_bn_running_var = self.L__mod___Mixed_6d_branch7x7dbl_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7dbl_4_bn_weight = self.L__mod___Mixed_6d_branch7x7dbl_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7dbl_4_bn_bias = self.L__mod___Mixed_6d_branch7x7dbl_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_299 = torch.nn.functional.batch_norm(x_298, l__mod___mixed_6d_branch7x7dbl_4_bn_running_mean, l__mod___mixed_6d_branch7x7dbl_4_bn_running_var, l__mod___mixed_6d_branch7x7dbl_4_bn_weight, l__mod___mixed_6d_branch7x7dbl_4_bn_bias, True, 0.1, 0.001);  x_298 = l__mod___mixed_6d_branch7x7dbl_4_bn_running_mean = l__mod___mixed_6d_branch7x7dbl_4_bn_running_var = l__mod___mixed_6d_branch7x7dbl_4_bn_weight = l__mod___mixed_6d_branch7x7dbl_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_300 = self.L__mod___Mixed_6d_branch7x7dbl_4_bn_drop(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_13 = self.L__mod___Mixed_6d_branch7x7dbl_4_bn_act(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_303 = self.L__mod___Mixed_6d_branch7x7dbl_5_conv(branch7x7dbl_13);  branch7x7dbl_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch7x7dbl_5_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch7x7dbl_5_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__58 = l__mod___mixed_6d_branch7x7dbl_5_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch7x7dbl_5_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_5_bn_running_mean = self.L__mod___Mixed_6d_branch7x7dbl_5_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch7x7dbl_5_bn_running_var = self.L__mod___Mixed_6d_branch7x7dbl_5_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch7x7dbl_5_bn_weight = self.L__mod___Mixed_6d_branch7x7dbl_5_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch7x7dbl_5_bn_bias = self.L__mod___Mixed_6d_branch7x7dbl_5_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_304 = torch.nn.functional.batch_norm(x_303, l__mod___mixed_6d_branch7x7dbl_5_bn_running_mean, l__mod___mixed_6d_branch7x7dbl_5_bn_running_var, l__mod___mixed_6d_branch7x7dbl_5_bn_weight, l__mod___mixed_6d_branch7x7dbl_5_bn_bias, True, 0.1, 0.001);  x_303 = l__mod___mixed_6d_branch7x7dbl_5_bn_running_mean = l__mod___mixed_6d_branch7x7dbl_5_bn_running_var = l__mod___mixed_6d_branch7x7dbl_5_bn_weight = l__mod___mixed_6d_branch7x7dbl_5_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_305 = self.L__mod___Mixed_6d_branch7x7dbl_5_bn_drop(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_14 = self.L__mod___Mixed_6d_branch7x7dbl_5_bn_act(x_305);  x_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_11 = torch._C._nn.avg_pool2d(x_262, kernel_size = 3, stride = 1, padding = 1);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_308 = self.L__mod___Mixed_6d_branch_pool_conv(branch_pool_11);  branch_pool_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6d_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_6d_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__59 = l__mod___mixed_6d_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_6d_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch_pool_bn_running_mean = self.L__mod___Mixed_6d_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6d_branch_pool_bn_running_var = self.L__mod___Mixed_6d_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6d_branch_pool_bn_weight = self.L__mod___Mixed_6d_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6d_branch_pool_bn_bias = self.L__mod___Mixed_6d_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_309 = torch.nn.functional.batch_norm(x_308, l__mod___mixed_6d_branch_pool_bn_running_mean, l__mod___mixed_6d_branch_pool_bn_running_var, l__mod___mixed_6d_branch_pool_bn_weight, l__mod___mixed_6d_branch_pool_bn_bias, True, 0.1, 0.001);  x_308 = l__mod___mixed_6d_branch_pool_bn_running_mean = l__mod___mixed_6d_branch_pool_bn_running_var = l__mod___mixed_6d_branch_pool_bn_weight = l__mod___mixed_6d_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_310 = self.L__mod___Mixed_6d_branch_pool_bn_drop(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_12 = self.L__mod___Mixed_6d_branch_pool_bn_act(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    x_313 = torch.cat([branch1x1_5, branch7x7_8, branch7x7dbl_14, branch_pool_12], 1);  branch1x1_5 = branch7x7_8 = branch7x7dbl_14 = branch_pool_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_314 = self.L__mod___Mixed_6e_branch1x1_conv(x_313)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__60 = l__mod___mixed_6e_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch1x1_bn_running_mean = self.L__mod___Mixed_6e_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch1x1_bn_running_var = self.L__mod___Mixed_6e_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch1x1_bn_weight = self.L__mod___Mixed_6e_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch1x1_bn_bias = self.L__mod___Mixed_6e_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_315 = torch.nn.functional.batch_norm(x_314, l__mod___mixed_6e_branch1x1_bn_running_mean, l__mod___mixed_6e_branch1x1_bn_running_var, l__mod___mixed_6e_branch1x1_bn_weight, l__mod___mixed_6e_branch1x1_bn_bias, True, 0.1, 0.001);  x_314 = l__mod___mixed_6e_branch1x1_bn_running_mean = l__mod___mixed_6e_branch1x1_bn_running_var = l__mod___mixed_6e_branch1x1_bn_weight = l__mod___mixed_6e_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_316 = self.L__mod___Mixed_6e_branch1x1_bn_drop(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_6 = self.L__mod___Mixed_6e_branch1x1_bn_act(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_319 = self.L__mod___Mixed_6e_branch7x7_1_conv(x_313)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7_1_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__61 = l__mod___mixed_6e_branch7x7_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7_1_bn_running_mean = self.L__mod___Mixed_6e_branch7x7_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7_1_bn_running_var = self.L__mod___Mixed_6e_branch7x7_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7_1_bn_weight = self.L__mod___Mixed_6e_branch7x7_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7_1_bn_bias = self.L__mod___Mixed_6e_branch7x7_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_320 = torch.nn.functional.batch_norm(x_319, l__mod___mixed_6e_branch7x7_1_bn_running_mean, l__mod___mixed_6e_branch7x7_1_bn_running_var, l__mod___mixed_6e_branch7x7_1_bn_weight, l__mod___mixed_6e_branch7x7_1_bn_bias, True, 0.1, 0.001);  x_319 = l__mod___mixed_6e_branch7x7_1_bn_running_mean = l__mod___mixed_6e_branch7x7_1_bn_running_var = l__mod___mixed_6e_branch7x7_1_bn_weight = l__mod___mixed_6e_branch7x7_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_321 = self.L__mod___Mixed_6e_branch7x7_1_bn_drop(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_9 = self.L__mod___Mixed_6e_branch7x7_1_bn_act(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_324 = self.L__mod___Mixed_6e_branch7x7_2_conv(branch7x7_9);  branch7x7_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7_2_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__62 = l__mod___mixed_6e_branch7x7_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7_2_bn_running_mean = self.L__mod___Mixed_6e_branch7x7_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7_2_bn_running_var = self.L__mod___Mixed_6e_branch7x7_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7_2_bn_weight = self.L__mod___Mixed_6e_branch7x7_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7_2_bn_bias = self.L__mod___Mixed_6e_branch7x7_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_325 = torch.nn.functional.batch_norm(x_324, l__mod___mixed_6e_branch7x7_2_bn_running_mean, l__mod___mixed_6e_branch7x7_2_bn_running_var, l__mod___mixed_6e_branch7x7_2_bn_weight, l__mod___mixed_6e_branch7x7_2_bn_bias, True, 0.1, 0.001);  x_324 = l__mod___mixed_6e_branch7x7_2_bn_running_mean = l__mod___mixed_6e_branch7x7_2_bn_running_var = l__mod___mixed_6e_branch7x7_2_bn_weight = l__mod___mixed_6e_branch7x7_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_326 = self.L__mod___Mixed_6e_branch7x7_2_bn_drop(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_10 = self.L__mod___Mixed_6e_branch7x7_2_bn_act(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_329 = self.L__mod___Mixed_6e_branch7x7_3_conv(branch7x7_10);  branch7x7_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7_3_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__63 = l__mod___mixed_6e_branch7x7_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7_3_bn_running_mean = self.L__mod___Mixed_6e_branch7x7_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7_3_bn_running_var = self.L__mod___Mixed_6e_branch7x7_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7_3_bn_weight = self.L__mod___Mixed_6e_branch7x7_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7_3_bn_bias = self.L__mod___Mixed_6e_branch7x7_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_330 = torch.nn.functional.batch_norm(x_329, l__mod___mixed_6e_branch7x7_3_bn_running_mean, l__mod___mixed_6e_branch7x7_3_bn_running_var, l__mod___mixed_6e_branch7x7_3_bn_weight, l__mod___mixed_6e_branch7x7_3_bn_bias, True, 0.1, 0.001);  x_329 = l__mod___mixed_6e_branch7x7_3_bn_running_mean = l__mod___mixed_6e_branch7x7_3_bn_running_var = l__mod___mixed_6e_branch7x7_3_bn_weight = l__mod___mixed_6e_branch7x7_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_331 = self.L__mod___Mixed_6e_branch7x7_3_bn_drop(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7_11 = self.L__mod___Mixed_6e_branch7x7_3_bn_act(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_334 = self.L__mod___Mixed_6e_branch7x7dbl_1_conv(x_313)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__64 = l__mod___mixed_6e_branch7x7dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_1_bn_running_mean = self.L__mod___Mixed_6e_branch7x7dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_1_bn_running_var = self.L__mod___Mixed_6e_branch7x7dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7dbl_1_bn_weight = self.L__mod___Mixed_6e_branch7x7dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7dbl_1_bn_bias = self.L__mod___Mixed_6e_branch7x7dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_335 = torch.nn.functional.batch_norm(x_334, l__mod___mixed_6e_branch7x7dbl_1_bn_running_mean, l__mod___mixed_6e_branch7x7dbl_1_bn_running_var, l__mod___mixed_6e_branch7x7dbl_1_bn_weight, l__mod___mixed_6e_branch7x7dbl_1_bn_bias, True, 0.1, 0.001);  x_334 = l__mod___mixed_6e_branch7x7dbl_1_bn_running_mean = l__mod___mixed_6e_branch7x7dbl_1_bn_running_var = l__mod___mixed_6e_branch7x7dbl_1_bn_weight = l__mod___mixed_6e_branch7x7dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_336 = self.L__mod___Mixed_6e_branch7x7dbl_1_bn_drop(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_15 = self.L__mod___Mixed_6e_branch7x7dbl_1_bn_act(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_339 = self.L__mod___Mixed_6e_branch7x7dbl_2_conv(branch7x7dbl_15);  branch7x7dbl_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__65 = l__mod___mixed_6e_branch7x7dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_2_bn_running_mean = self.L__mod___Mixed_6e_branch7x7dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_2_bn_running_var = self.L__mod___Mixed_6e_branch7x7dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7dbl_2_bn_weight = self.L__mod___Mixed_6e_branch7x7dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7dbl_2_bn_bias = self.L__mod___Mixed_6e_branch7x7dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_340 = torch.nn.functional.batch_norm(x_339, l__mod___mixed_6e_branch7x7dbl_2_bn_running_mean, l__mod___mixed_6e_branch7x7dbl_2_bn_running_var, l__mod___mixed_6e_branch7x7dbl_2_bn_weight, l__mod___mixed_6e_branch7x7dbl_2_bn_bias, True, 0.1, 0.001);  x_339 = l__mod___mixed_6e_branch7x7dbl_2_bn_running_mean = l__mod___mixed_6e_branch7x7dbl_2_bn_running_var = l__mod___mixed_6e_branch7x7dbl_2_bn_weight = l__mod___mixed_6e_branch7x7dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_341 = self.L__mod___Mixed_6e_branch7x7dbl_2_bn_drop(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_16 = self.L__mod___Mixed_6e_branch7x7dbl_2_bn_act(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_344 = self.L__mod___Mixed_6e_branch7x7dbl_3_conv(branch7x7dbl_16);  branch7x7dbl_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7dbl_3_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7dbl_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__66 = l__mod___mixed_6e_branch7x7dbl_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7dbl_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_3_bn_running_mean = self.L__mod___Mixed_6e_branch7x7dbl_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_3_bn_running_var = self.L__mod___Mixed_6e_branch7x7dbl_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7dbl_3_bn_weight = self.L__mod___Mixed_6e_branch7x7dbl_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7dbl_3_bn_bias = self.L__mod___Mixed_6e_branch7x7dbl_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_345 = torch.nn.functional.batch_norm(x_344, l__mod___mixed_6e_branch7x7dbl_3_bn_running_mean, l__mod___mixed_6e_branch7x7dbl_3_bn_running_var, l__mod___mixed_6e_branch7x7dbl_3_bn_weight, l__mod___mixed_6e_branch7x7dbl_3_bn_bias, True, 0.1, 0.001);  x_344 = l__mod___mixed_6e_branch7x7dbl_3_bn_running_mean = l__mod___mixed_6e_branch7x7dbl_3_bn_running_var = l__mod___mixed_6e_branch7x7dbl_3_bn_weight = l__mod___mixed_6e_branch7x7dbl_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_346 = self.L__mod___Mixed_6e_branch7x7dbl_3_bn_drop(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_17 = self.L__mod___Mixed_6e_branch7x7dbl_3_bn_act(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_349 = self.L__mod___Mixed_6e_branch7x7dbl_4_conv(branch7x7dbl_17);  branch7x7dbl_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7dbl_4_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7dbl_4_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__67 = l__mod___mixed_6e_branch7x7dbl_4_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7dbl_4_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_4_bn_running_mean = self.L__mod___Mixed_6e_branch7x7dbl_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_4_bn_running_var = self.L__mod___Mixed_6e_branch7x7dbl_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7dbl_4_bn_weight = self.L__mod___Mixed_6e_branch7x7dbl_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7dbl_4_bn_bias = self.L__mod___Mixed_6e_branch7x7dbl_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_350 = torch.nn.functional.batch_norm(x_349, l__mod___mixed_6e_branch7x7dbl_4_bn_running_mean, l__mod___mixed_6e_branch7x7dbl_4_bn_running_var, l__mod___mixed_6e_branch7x7dbl_4_bn_weight, l__mod___mixed_6e_branch7x7dbl_4_bn_bias, True, 0.1, 0.001);  x_349 = l__mod___mixed_6e_branch7x7dbl_4_bn_running_mean = l__mod___mixed_6e_branch7x7dbl_4_bn_running_var = l__mod___mixed_6e_branch7x7dbl_4_bn_weight = l__mod___mixed_6e_branch7x7dbl_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_351 = self.L__mod___Mixed_6e_branch7x7dbl_4_bn_drop(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_18 = self.L__mod___Mixed_6e_branch7x7dbl_4_bn_act(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_354 = self.L__mod___Mixed_6e_branch7x7dbl_5_conv(branch7x7dbl_18);  branch7x7dbl_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch7x7dbl_5_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch7x7dbl_5_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__68 = l__mod___mixed_6e_branch7x7dbl_5_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch7x7dbl_5_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_5_bn_running_mean = self.L__mod___Mixed_6e_branch7x7dbl_5_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch7x7dbl_5_bn_running_var = self.L__mod___Mixed_6e_branch7x7dbl_5_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch7x7dbl_5_bn_weight = self.L__mod___Mixed_6e_branch7x7dbl_5_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch7x7dbl_5_bn_bias = self.L__mod___Mixed_6e_branch7x7dbl_5_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_355 = torch.nn.functional.batch_norm(x_354, l__mod___mixed_6e_branch7x7dbl_5_bn_running_mean, l__mod___mixed_6e_branch7x7dbl_5_bn_running_var, l__mod___mixed_6e_branch7x7dbl_5_bn_weight, l__mod___mixed_6e_branch7x7dbl_5_bn_bias, True, 0.1, 0.001);  x_354 = l__mod___mixed_6e_branch7x7dbl_5_bn_running_mean = l__mod___mixed_6e_branch7x7dbl_5_bn_running_var = l__mod___mixed_6e_branch7x7dbl_5_bn_weight = l__mod___mixed_6e_branch7x7dbl_5_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_356 = self.L__mod___Mixed_6e_branch7x7dbl_5_bn_drop(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7dbl_19 = self.L__mod___Mixed_6e_branch7x7dbl_5_bn_act(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_13 = torch._C._nn.avg_pool2d(x_313, kernel_size = 3, stride = 1, padding = 1);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_359 = self.L__mod___Mixed_6e_branch_pool_conv(branch_pool_13);  branch_pool_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_6e_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_6e_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__69 = l__mod___mixed_6e_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_6e_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch_pool_bn_running_mean = self.L__mod___Mixed_6e_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_6e_branch_pool_bn_running_var = self.L__mod___Mixed_6e_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_6e_branch_pool_bn_weight = self.L__mod___Mixed_6e_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_6e_branch_pool_bn_bias = self.L__mod___Mixed_6e_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_360 = torch.nn.functional.batch_norm(x_359, l__mod___mixed_6e_branch_pool_bn_running_mean, l__mod___mixed_6e_branch_pool_bn_running_var, l__mod___mixed_6e_branch_pool_bn_weight, l__mod___mixed_6e_branch_pool_bn_bias, True, 0.1, 0.001);  x_359 = l__mod___mixed_6e_branch_pool_bn_running_mean = l__mod___mixed_6e_branch_pool_bn_running_var = l__mod___mixed_6e_branch_pool_bn_weight = l__mod___mixed_6e_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_361 = self.L__mod___Mixed_6e_branch_pool_bn_drop(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_14 = self.L__mod___Mixed_6e_branch_pool_bn_act(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    x_365 = torch.cat([branch1x1_6, branch7x7_11, branch7x7dbl_19, branch_pool_14], 1);  branch1x1_6 = branch7x7_11 = branch7x7dbl_19 = branch_pool_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_366 = self.L__mod___Mixed_7a_branch3x3_1_conv(x_365)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7a_branch3x3_1_bn_num_batches_tracked = self.L__mod___Mixed_7a_branch3x3_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__70 = l__mod___mixed_7a_branch3x3_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7a_branch3x3_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch3x3_1_bn_running_mean = self.L__mod___Mixed_7a_branch3x3_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch3x3_1_bn_running_var = self.L__mod___Mixed_7a_branch3x3_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7a_branch3x3_1_bn_weight = self.L__mod___Mixed_7a_branch3x3_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7a_branch3x3_1_bn_bias = self.L__mod___Mixed_7a_branch3x3_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_367 = torch.nn.functional.batch_norm(x_366, l__mod___mixed_7a_branch3x3_1_bn_running_mean, l__mod___mixed_7a_branch3x3_1_bn_running_var, l__mod___mixed_7a_branch3x3_1_bn_weight, l__mod___mixed_7a_branch3x3_1_bn_bias, True, 0.1, 0.001);  x_366 = l__mod___mixed_7a_branch3x3_1_bn_running_mean = l__mod___mixed_7a_branch3x3_1_bn_running_var = l__mod___mixed_7a_branch3x3_1_bn_weight = l__mod___mixed_7a_branch3x3_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_368 = self.L__mod___Mixed_7a_branch3x3_1_bn_drop(x_367);  x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3_1 = self.L__mod___Mixed_7a_branch3x3_1_bn_act(x_368);  x_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_371 = self.L__mod___Mixed_7a_branch3x3_2_conv(branch3x3_1);  branch3x3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7a_branch3x3_2_bn_num_batches_tracked = self.L__mod___Mixed_7a_branch3x3_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__71 = l__mod___mixed_7a_branch3x3_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_7a_branch3x3_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch3x3_2_bn_running_mean = self.L__mod___Mixed_7a_branch3x3_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch3x3_2_bn_running_var = self.L__mod___Mixed_7a_branch3x3_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7a_branch3x3_2_bn_weight = self.L__mod___Mixed_7a_branch3x3_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7a_branch3x3_2_bn_bias = self.L__mod___Mixed_7a_branch3x3_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_372 = torch.nn.functional.batch_norm(x_371, l__mod___mixed_7a_branch3x3_2_bn_running_mean, l__mod___mixed_7a_branch3x3_2_bn_running_var, l__mod___mixed_7a_branch3x3_2_bn_weight, l__mod___mixed_7a_branch3x3_2_bn_bias, True, 0.1, 0.001);  x_371 = l__mod___mixed_7a_branch3x3_2_bn_running_mean = l__mod___mixed_7a_branch3x3_2_bn_running_var = l__mod___mixed_7a_branch3x3_2_bn_weight = l__mod___mixed_7a_branch3x3_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_373 = self.L__mod___Mixed_7a_branch3x3_2_bn_drop(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3_2 = self.L__mod___Mixed_7a_branch3x3_2_bn_act(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_376 = self.L__mod___Mixed_7a_branch7x7x3_1_conv(x_365)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7a_branch7x7x3_1_bn_num_batches_tracked = self.L__mod___Mixed_7a_branch7x7x3_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__72 = l__mod___mixed_7a_branch7x7x3_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7a_branch7x7x3_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_1_bn_running_mean = self.L__mod___Mixed_7a_branch7x7x3_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_1_bn_running_var = self.L__mod___Mixed_7a_branch7x7x3_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7a_branch7x7x3_1_bn_weight = self.L__mod___Mixed_7a_branch7x7x3_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7a_branch7x7x3_1_bn_bias = self.L__mod___Mixed_7a_branch7x7x3_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_377 = torch.nn.functional.batch_norm(x_376, l__mod___mixed_7a_branch7x7x3_1_bn_running_mean, l__mod___mixed_7a_branch7x7x3_1_bn_running_var, l__mod___mixed_7a_branch7x7x3_1_bn_weight, l__mod___mixed_7a_branch7x7x3_1_bn_bias, True, 0.1, 0.001);  x_376 = l__mod___mixed_7a_branch7x7x3_1_bn_running_mean = l__mod___mixed_7a_branch7x7x3_1_bn_running_var = l__mod___mixed_7a_branch7x7x3_1_bn_weight = l__mod___mixed_7a_branch7x7x3_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_378 = self.L__mod___Mixed_7a_branch7x7x3_1_bn_drop(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7x3 = self.L__mod___Mixed_7a_branch7x7x3_1_bn_act(x_378);  x_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_381 = self.L__mod___Mixed_7a_branch7x7x3_2_conv(branch7x7x3);  branch7x7x3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7a_branch7x7x3_2_bn_num_batches_tracked = self.L__mod___Mixed_7a_branch7x7x3_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__73 = l__mod___mixed_7a_branch7x7x3_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_7a_branch7x7x3_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_2_bn_running_mean = self.L__mod___Mixed_7a_branch7x7x3_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_2_bn_running_var = self.L__mod___Mixed_7a_branch7x7x3_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7a_branch7x7x3_2_bn_weight = self.L__mod___Mixed_7a_branch7x7x3_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7a_branch7x7x3_2_bn_bias = self.L__mod___Mixed_7a_branch7x7x3_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_382 = torch.nn.functional.batch_norm(x_381, l__mod___mixed_7a_branch7x7x3_2_bn_running_mean, l__mod___mixed_7a_branch7x7x3_2_bn_running_var, l__mod___mixed_7a_branch7x7x3_2_bn_weight, l__mod___mixed_7a_branch7x7x3_2_bn_bias, True, 0.1, 0.001);  x_381 = l__mod___mixed_7a_branch7x7x3_2_bn_running_mean = l__mod___mixed_7a_branch7x7x3_2_bn_running_var = l__mod___mixed_7a_branch7x7x3_2_bn_weight = l__mod___mixed_7a_branch7x7x3_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_383 = self.L__mod___Mixed_7a_branch7x7x3_2_bn_drop(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7x3_1 = self.L__mod___Mixed_7a_branch7x7x3_2_bn_act(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_386 = self.L__mod___Mixed_7a_branch7x7x3_3_conv(branch7x7x3_1);  branch7x7x3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7a_branch7x7x3_3_bn_num_batches_tracked = self.L__mod___Mixed_7a_branch7x7x3_3_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__74 = l__mod___mixed_7a_branch7x7x3_3_bn_num_batches_tracked.add_(1);  l__mod___mixed_7a_branch7x7x3_3_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_3_bn_running_mean = self.L__mod___Mixed_7a_branch7x7x3_3_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_3_bn_running_var = self.L__mod___Mixed_7a_branch7x7x3_3_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7a_branch7x7x3_3_bn_weight = self.L__mod___Mixed_7a_branch7x7x3_3_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7a_branch7x7x3_3_bn_bias = self.L__mod___Mixed_7a_branch7x7x3_3_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_387 = torch.nn.functional.batch_norm(x_386, l__mod___mixed_7a_branch7x7x3_3_bn_running_mean, l__mod___mixed_7a_branch7x7x3_3_bn_running_var, l__mod___mixed_7a_branch7x7x3_3_bn_weight, l__mod___mixed_7a_branch7x7x3_3_bn_bias, True, 0.1, 0.001);  x_386 = l__mod___mixed_7a_branch7x7x3_3_bn_running_mean = l__mod___mixed_7a_branch7x7x3_3_bn_running_var = l__mod___mixed_7a_branch7x7x3_3_bn_weight = l__mod___mixed_7a_branch7x7x3_3_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_388 = self.L__mod___Mixed_7a_branch7x7x3_3_bn_drop(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7x3_2 = self.L__mod___Mixed_7a_branch7x7x3_3_bn_act(x_388);  x_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_391 = self.L__mod___Mixed_7a_branch7x7x3_4_conv(branch7x7x3_2);  branch7x7x3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7a_branch7x7x3_4_bn_num_batches_tracked = self.L__mod___Mixed_7a_branch7x7x3_4_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__75 = l__mod___mixed_7a_branch7x7x3_4_bn_num_batches_tracked.add_(1);  l__mod___mixed_7a_branch7x7x3_4_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_4_bn_running_mean = self.L__mod___Mixed_7a_branch7x7x3_4_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7a_branch7x7x3_4_bn_running_var = self.L__mod___Mixed_7a_branch7x7x3_4_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7a_branch7x7x3_4_bn_weight = self.L__mod___Mixed_7a_branch7x7x3_4_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7a_branch7x7x3_4_bn_bias = self.L__mod___Mixed_7a_branch7x7x3_4_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_392 = torch.nn.functional.batch_norm(x_391, l__mod___mixed_7a_branch7x7x3_4_bn_running_mean, l__mod___mixed_7a_branch7x7x3_4_bn_running_var, l__mod___mixed_7a_branch7x7x3_4_bn_weight, l__mod___mixed_7a_branch7x7x3_4_bn_bias, True, 0.1, 0.001);  x_391 = l__mod___mixed_7a_branch7x7x3_4_bn_running_mean = l__mod___mixed_7a_branch7x7x3_4_bn_running_var = l__mod___mixed_7a_branch7x7x3_4_bn_weight = l__mod___mixed_7a_branch7x7x3_4_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_393 = self.L__mod___Mixed_7a_branch7x7x3_4_bn_drop(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch7x7x3_3 = self.L__mod___Mixed_7a_branch7x7x3_4_bn_act(x_393);  x_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:153, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
    branch_pool_15 = torch.nn.functional.max_pool2d(x_365, kernel_size = 3, stride = 2);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:159, code: return torch.cat(outputs, 1)
    x_396 = torch.cat([branch3x3_2, branch7x7x3_3, branch_pool_15], 1);  branch3x3_2 = branch7x7x3_3 = branch_pool_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_397 = self.L__mod___Mixed_7b_branch1x1_conv(x_396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__76 = l__mod___mixed_7b_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch1x1_bn_running_mean = self.L__mod___Mixed_7b_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch1x1_bn_running_var = self.L__mod___Mixed_7b_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch1x1_bn_weight = self.L__mod___Mixed_7b_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch1x1_bn_bias = self.L__mod___Mixed_7b_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_398 = torch.nn.functional.batch_norm(x_397, l__mod___mixed_7b_branch1x1_bn_running_mean, l__mod___mixed_7b_branch1x1_bn_running_var, l__mod___mixed_7b_branch1x1_bn_weight, l__mod___mixed_7b_branch1x1_bn_bias, True, 0.1, 0.001);  x_397 = l__mod___mixed_7b_branch1x1_bn_running_mean = l__mod___mixed_7b_branch1x1_bn_running_var = l__mod___mixed_7b_branch1x1_bn_weight = l__mod___mixed_7b_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_399 = self.L__mod___Mixed_7b_branch1x1_bn_drop(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_7 = self.L__mod___Mixed_7b_branch1x1_bn_act(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_402 = self.L__mod___Mixed_7b_branch3x3_1_conv(x_396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch3x3_1_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch3x3_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__77 = l__mod___mixed_7b_branch3x3_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch3x3_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3_1_bn_running_mean = self.L__mod___Mixed_7b_branch3x3_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3_1_bn_running_var = self.L__mod___Mixed_7b_branch3x3_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch3x3_1_bn_weight = self.L__mod___Mixed_7b_branch3x3_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch3x3_1_bn_bias = self.L__mod___Mixed_7b_branch3x3_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_403 = torch.nn.functional.batch_norm(x_402, l__mod___mixed_7b_branch3x3_1_bn_running_mean, l__mod___mixed_7b_branch3x3_1_bn_running_var, l__mod___mixed_7b_branch3x3_1_bn_weight, l__mod___mixed_7b_branch3x3_1_bn_bias, True, 0.1, 0.001);  x_402 = l__mod___mixed_7b_branch3x3_1_bn_running_mean = l__mod___mixed_7b_branch3x3_1_bn_running_var = l__mod___mixed_7b_branch3x3_1_bn_weight = l__mod___mixed_7b_branch3x3_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_404 = self.L__mod___Mixed_7b_branch3x3_1_bn_drop(x_403);  x_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3_3 = self.L__mod___Mixed_7b_branch3x3_1_bn_act(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_407 = self.L__mod___Mixed_7b_branch3x3_2a_conv(branch3x3_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch3x3_2a_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch3x3_2a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__78 = l__mod___mixed_7b_branch3x3_2a_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch3x3_2a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3_2a_bn_running_mean = self.L__mod___Mixed_7b_branch3x3_2a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3_2a_bn_running_var = self.L__mod___Mixed_7b_branch3x3_2a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch3x3_2a_bn_weight = self.L__mod___Mixed_7b_branch3x3_2a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch3x3_2a_bn_bias = self.L__mod___Mixed_7b_branch3x3_2a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_408 = torch.nn.functional.batch_norm(x_407, l__mod___mixed_7b_branch3x3_2a_bn_running_mean, l__mod___mixed_7b_branch3x3_2a_bn_running_var, l__mod___mixed_7b_branch3x3_2a_bn_weight, l__mod___mixed_7b_branch3x3_2a_bn_bias, True, 0.1, 0.001);  x_407 = l__mod___mixed_7b_branch3x3_2a_bn_running_mean = l__mod___mixed_7b_branch3x3_2a_bn_running_var = l__mod___mixed_7b_branch3x3_2a_bn_weight = l__mod___mixed_7b_branch3x3_2a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_409 = self.L__mod___Mixed_7b_branch3x3_2a_bn_drop(x_408);  x_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_411 = self.L__mod___Mixed_7b_branch3x3_2a_bn_act(x_409);  x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_412 = self.L__mod___Mixed_7b_branch3x3_2b_conv(branch3x3_3);  branch3x3_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch3x3_2b_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch3x3_2b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__79 = l__mod___mixed_7b_branch3x3_2b_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch3x3_2b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3_2b_bn_running_mean = self.L__mod___Mixed_7b_branch3x3_2b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3_2b_bn_running_var = self.L__mod___Mixed_7b_branch3x3_2b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch3x3_2b_bn_weight = self.L__mod___Mixed_7b_branch3x3_2b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch3x3_2b_bn_bias = self.L__mod___Mixed_7b_branch3x3_2b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_413 = torch.nn.functional.batch_norm(x_412, l__mod___mixed_7b_branch3x3_2b_bn_running_mean, l__mod___mixed_7b_branch3x3_2b_bn_running_var, l__mod___mixed_7b_branch3x3_2b_bn_weight, l__mod___mixed_7b_branch3x3_2b_bn_bias, True, 0.1, 0.001);  x_412 = l__mod___mixed_7b_branch3x3_2b_bn_running_mean = l__mod___mixed_7b_branch3x3_2b_bn_running_var = l__mod___mixed_7b_branch3x3_2b_bn_weight = l__mod___mixed_7b_branch3x3_2b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_414 = self.L__mod___Mixed_7b_branch3x3_2b_bn_drop(x_413);  x_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_416 = self.L__mod___Mixed_7b_branch3x3_2b_bn_act(x_414);  x_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:188, code: branch3x3 = torch.cat(branch3x3, 1)
    branch3x3_4 = torch.cat([x_411, x_416], 1);  x_411 = x_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_417 = self.L__mod___Mixed_7b_branch3x3dbl_1_conv(x_396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch3x3dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch3x3dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__80 = l__mod___mixed_7b_branch3x3dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch3x3dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_1_bn_running_mean = self.L__mod___Mixed_7b_branch3x3dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_1_bn_running_var = self.L__mod___Mixed_7b_branch3x3dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch3x3dbl_1_bn_weight = self.L__mod___Mixed_7b_branch3x3dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch3x3dbl_1_bn_bias = self.L__mod___Mixed_7b_branch3x3dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_418 = torch.nn.functional.batch_norm(x_417, l__mod___mixed_7b_branch3x3dbl_1_bn_running_mean, l__mod___mixed_7b_branch3x3dbl_1_bn_running_var, l__mod___mixed_7b_branch3x3dbl_1_bn_weight, l__mod___mixed_7b_branch3x3dbl_1_bn_bias, True, 0.1, 0.001);  x_417 = l__mod___mixed_7b_branch3x3dbl_1_bn_running_mean = l__mod___mixed_7b_branch3x3dbl_1_bn_running_var = l__mod___mixed_7b_branch3x3dbl_1_bn_weight = l__mod___mixed_7b_branch3x3dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_419 = self.L__mod___Mixed_7b_branch3x3dbl_1_bn_drop(x_418);  x_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_12 = self.L__mod___Mixed_7b_branch3x3dbl_1_bn_act(x_419);  x_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_422 = self.L__mod___Mixed_7b_branch3x3dbl_2_conv(branch3x3dbl_12);  branch3x3dbl_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch3x3dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch3x3dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__81 = l__mod___mixed_7b_branch3x3dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch3x3dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_2_bn_running_mean = self.L__mod___Mixed_7b_branch3x3dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_2_bn_running_var = self.L__mod___Mixed_7b_branch3x3dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch3x3dbl_2_bn_weight = self.L__mod___Mixed_7b_branch3x3dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch3x3dbl_2_bn_bias = self.L__mod___Mixed_7b_branch3x3dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_423 = torch.nn.functional.batch_norm(x_422, l__mod___mixed_7b_branch3x3dbl_2_bn_running_mean, l__mod___mixed_7b_branch3x3dbl_2_bn_running_var, l__mod___mixed_7b_branch3x3dbl_2_bn_weight, l__mod___mixed_7b_branch3x3dbl_2_bn_bias, True, 0.1, 0.001);  x_422 = l__mod___mixed_7b_branch3x3dbl_2_bn_running_mean = l__mod___mixed_7b_branch3x3dbl_2_bn_running_var = l__mod___mixed_7b_branch3x3dbl_2_bn_weight = l__mod___mixed_7b_branch3x3dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_424 = self.L__mod___Mixed_7b_branch3x3dbl_2_bn_drop(x_423);  x_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_13 = self.L__mod___Mixed_7b_branch3x3dbl_2_bn_act(x_424);  x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_427 = self.L__mod___Mixed_7b_branch3x3dbl_3a_conv(branch3x3dbl_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch3x3dbl_3a_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch3x3dbl_3a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__82 = l__mod___mixed_7b_branch3x3dbl_3a_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch3x3dbl_3a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_3a_bn_running_mean = self.L__mod___Mixed_7b_branch3x3dbl_3a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_3a_bn_running_var = self.L__mod___Mixed_7b_branch3x3dbl_3a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch3x3dbl_3a_bn_weight = self.L__mod___Mixed_7b_branch3x3dbl_3a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch3x3dbl_3a_bn_bias = self.L__mod___Mixed_7b_branch3x3dbl_3a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_428 = torch.nn.functional.batch_norm(x_427, l__mod___mixed_7b_branch3x3dbl_3a_bn_running_mean, l__mod___mixed_7b_branch3x3dbl_3a_bn_running_var, l__mod___mixed_7b_branch3x3dbl_3a_bn_weight, l__mod___mixed_7b_branch3x3dbl_3a_bn_bias, True, 0.1, 0.001);  x_427 = l__mod___mixed_7b_branch3x3dbl_3a_bn_running_mean = l__mod___mixed_7b_branch3x3dbl_3a_bn_running_var = l__mod___mixed_7b_branch3x3dbl_3a_bn_weight = l__mod___mixed_7b_branch3x3dbl_3a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_429 = self.L__mod___Mixed_7b_branch3x3dbl_3a_bn_drop(x_428);  x_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_431 = self.L__mod___Mixed_7b_branch3x3dbl_3a_bn_act(x_429);  x_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_432 = self.L__mod___Mixed_7b_branch3x3dbl_3b_conv(branch3x3dbl_13);  branch3x3dbl_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch3x3dbl_3b_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch3x3dbl_3b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__83 = l__mod___mixed_7b_branch3x3dbl_3b_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch3x3dbl_3b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_3b_bn_running_mean = self.L__mod___Mixed_7b_branch3x3dbl_3b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch3x3dbl_3b_bn_running_var = self.L__mod___Mixed_7b_branch3x3dbl_3b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch3x3dbl_3b_bn_weight = self.L__mod___Mixed_7b_branch3x3dbl_3b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch3x3dbl_3b_bn_bias = self.L__mod___Mixed_7b_branch3x3dbl_3b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_433 = torch.nn.functional.batch_norm(x_432, l__mod___mixed_7b_branch3x3dbl_3b_bn_running_mean, l__mod___mixed_7b_branch3x3dbl_3b_bn_running_var, l__mod___mixed_7b_branch3x3dbl_3b_bn_weight, l__mod___mixed_7b_branch3x3dbl_3b_bn_bias, True, 0.1, 0.001);  x_432 = l__mod___mixed_7b_branch3x3dbl_3b_bn_running_mean = l__mod___mixed_7b_branch3x3dbl_3b_bn_running_var = l__mod___mixed_7b_branch3x3dbl_3b_bn_weight = l__mod___mixed_7b_branch3x3dbl_3b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_434 = self.L__mod___Mixed_7b_branch3x3dbl_3b_bn_drop(x_433);  x_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_436 = self.L__mod___Mixed_7b_branch3x3dbl_3b_bn_act(x_434);  x_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:196, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
    branch3x3dbl_14 = torch.cat([x_431, x_436], 1);  x_431 = x_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:198, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_16 = torch._C._nn.avg_pool2d(x_396, kernel_size = 3, stride = 1, padding = 1);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_437 = self.L__mod___Mixed_7b_branch_pool_conv(branch_pool_16);  branch_pool_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7b_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_7b_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__84 = l__mod___mixed_7b_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_7b_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch_pool_bn_running_mean = self.L__mod___Mixed_7b_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7b_branch_pool_bn_running_var = self.L__mod___Mixed_7b_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7b_branch_pool_bn_weight = self.L__mod___Mixed_7b_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7b_branch_pool_bn_bias = self.L__mod___Mixed_7b_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_438 = torch.nn.functional.batch_norm(x_437, l__mod___mixed_7b_branch_pool_bn_running_mean, l__mod___mixed_7b_branch_pool_bn_running_var, l__mod___mixed_7b_branch_pool_bn_weight, l__mod___mixed_7b_branch_pool_bn_bias, True, 0.1, 0.001);  x_437 = l__mod___mixed_7b_branch_pool_bn_running_mean = l__mod___mixed_7b_branch_pool_bn_running_var = l__mod___mixed_7b_branch_pool_bn_weight = l__mod___mixed_7b_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_439 = self.L__mod___Mixed_7b_branch_pool_bn_drop(x_438);  x_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_17 = self.L__mod___Mixed_7b_branch_pool_bn_act(x_439);  x_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:206, code: return torch.cat(outputs, 1)
    x_442 = torch.cat([branch1x1_7, branch3x3_4, branch3x3dbl_14, branch_pool_17], 1);  branch1x1_7 = branch3x3_4 = branch3x3dbl_14 = branch_pool_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_443 = self.L__mod___Mixed_7c_branch1x1_conv(x_442)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch1x1_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch1x1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__85 = l__mod___mixed_7c_branch1x1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch1x1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch1x1_bn_running_mean = self.L__mod___Mixed_7c_branch1x1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch1x1_bn_running_var = self.L__mod___Mixed_7c_branch1x1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch1x1_bn_weight = self.L__mod___Mixed_7c_branch1x1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch1x1_bn_bias = self.L__mod___Mixed_7c_branch1x1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_444 = torch.nn.functional.batch_norm(x_443, l__mod___mixed_7c_branch1x1_bn_running_mean, l__mod___mixed_7c_branch1x1_bn_running_var, l__mod___mixed_7c_branch1x1_bn_weight, l__mod___mixed_7c_branch1x1_bn_bias, True, 0.1, 0.001);  x_443 = l__mod___mixed_7c_branch1x1_bn_running_mean = l__mod___mixed_7c_branch1x1_bn_running_var = l__mod___mixed_7c_branch1x1_bn_weight = l__mod___mixed_7c_branch1x1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_445 = self.L__mod___Mixed_7c_branch1x1_bn_drop(x_444);  x_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch1x1_8 = self.L__mod___Mixed_7c_branch1x1_bn_act(x_445);  x_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_448 = self.L__mod___Mixed_7c_branch3x3_1_conv(x_442)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch3x3_1_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch3x3_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__86 = l__mod___mixed_7c_branch3x3_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch3x3_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3_1_bn_running_mean = self.L__mod___Mixed_7c_branch3x3_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3_1_bn_running_var = self.L__mod___Mixed_7c_branch3x3_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch3x3_1_bn_weight = self.L__mod___Mixed_7c_branch3x3_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch3x3_1_bn_bias = self.L__mod___Mixed_7c_branch3x3_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_449 = torch.nn.functional.batch_norm(x_448, l__mod___mixed_7c_branch3x3_1_bn_running_mean, l__mod___mixed_7c_branch3x3_1_bn_running_var, l__mod___mixed_7c_branch3x3_1_bn_weight, l__mod___mixed_7c_branch3x3_1_bn_bias, True, 0.1, 0.001);  x_448 = l__mod___mixed_7c_branch3x3_1_bn_running_mean = l__mod___mixed_7c_branch3x3_1_bn_running_var = l__mod___mixed_7c_branch3x3_1_bn_weight = l__mod___mixed_7c_branch3x3_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_450 = self.L__mod___Mixed_7c_branch3x3_1_bn_drop(x_449);  x_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3_5 = self.L__mod___Mixed_7c_branch3x3_1_bn_act(x_450);  x_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_453 = self.L__mod___Mixed_7c_branch3x3_2a_conv(branch3x3_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch3x3_2a_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch3x3_2a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__87 = l__mod___mixed_7c_branch3x3_2a_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch3x3_2a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3_2a_bn_running_mean = self.L__mod___Mixed_7c_branch3x3_2a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3_2a_bn_running_var = self.L__mod___Mixed_7c_branch3x3_2a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch3x3_2a_bn_weight = self.L__mod___Mixed_7c_branch3x3_2a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch3x3_2a_bn_bias = self.L__mod___Mixed_7c_branch3x3_2a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_454 = torch.nn.functional.batch_norm(x_453, l__mod___mixed_7c_branch3x3_2a_bn_running_mean, l__mod___mixed_7c_branch3x3_2a_bn_running_var, l__mod___mixed_7c_branch3x3_2a_bn_weight, l__mod___mixed_7c_branch3x3_2a_bn_bias, True, 0.1, 0.001);  x_453 = l__mod___mixed_7c_branch3x3_2a_bn_running_mean = l__mod___mixed_7c_branch3x3_2a_bn_running_var = l__mod___mixed_7c_branch3x3_2a_bn_weight = l__mod___mixed_7c_branch3x3_2a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_455 = self.L__mod___Mixed_7c_branch3x3_2a_bn_drop(x_454);  x_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_457 = self.L__mod___Mixed_7c_branch3x3_2a_bn_act(x_455);  x_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_458 = self.L__mod___Mixed_7c_branch3x3_2b_conv(branch3x3_5);  branch3x3_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch3x3_2b_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch3x3_2b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__88 = l__mod___mixed_7c_branch3x3_2b_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch3x3_2b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3_2b_bn_running_mean = self.L__mod___Mixed_7c_branch3x3_2b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3_2b_bn_running_var = self.L__mod___Mixed_7c_branch3x3_2b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch3x3_2b_bn_weight = self.L__mod___Mixed_7c_branch3x3_2b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch3x3_2b_bn_bias = self.L__mod___Mixed_7c_branch3x3_2b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_459 = torch.nn.functional.batch_norm(x_458, l__mod___mixed_7c_branch3x3_2b_bn_running_mean, l__mod___mixed_7c_branch3x3_2b_bn_running_var, l__mod___mixed_7c_branch3x3_2b_bn_weight, l__mod___mixed_7c_branch3x3_2b_bn_bias, True, 0.1, 0.001);  x_458 = l__mod___mixed_7c_branch3x3_2b_bn_running_mean = l__mod___mixed_7c_branch3x3_2b_bn_running_var = l__mod___mixed_7c_branch3x3_2b_bn_weight = l__mod___mixed_7c_branch3x3_2b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_460 = self.L__mod___Mixed_7c_branch3x3_2b_bn_drop(x_459);  x_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_462 = self.L__mod___Mixed_7c_branch3x3_2b_bn_act(x_460);  x_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:188, code: branch3x3 = torch.cat(branch3x3, 1)
    branch3x3_6 = torch.cat([x_457, x_462], 1);  x_457 = x_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_463 = self.L__mod___Mixed_7c_branch3x3dbl_1_conv(x_442)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch3x3dbl_1_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch3x3dbl_1_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__89 = l__mod___mixed_7c_branch3x3dbl_1_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch3x3dbl_1_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_1_bn_running_mean = self.L__mod___Mixed_7c_branch3x3dbl_1_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_1_bn_running_var = self.L__mod___Mixed_7c_branch3x3dbl_1_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch3x3dbl_1_bn_weight = self.L__mod___Mixed_7c_branch3x3dbl_1_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch3x3dbl_1_bn_bias = self.L__mod___Mixed_7c_branch3x3dbl_1_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_464 = torch.nn.functional.batch_norm(x_463, l__mod___mixed_7c_branch3x3dbl_1_bn_running_mean, l__mod___mixed_7c_branch3x3dbl_1_bn_running_var, l__mod___mixed_7c_branch3x3dbl_1_bn_weight, l__mod___mixed_7c_branch3x3dbl_1_bn_bias, True, 0.1, 0.001);  x_463 = l__mod___mixed_7c_branch3x3dbl_1_bn_running_mean = l__mod___mixed_7c_branch3x3dbl_1_bn_running_var = l__mod___mixed_7c_branch3x3dbl_1_bn_weight = l__mod___mixed_7c_branch3x3dbl_1_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_465 = self.L__mod___Mixed_7c_branch3x3dbl_1_bn_drop(x_464);  x_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_15 = self.L__mod___Mixed_7c_branch3x3dbl_1_bn_act(x_465);  x_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_468 = self.L__mod___Mixed_7c_branch3x3dbl_2_conv(branch3x3dbl_15);  branch3x3dbl_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch3x3dbl_2_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch3x3dbl_2_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__90 = l__mod___mixed_7c_branch3x3dbl_2_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch3x3dbl_2_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_2_bn_running_mean = self.L__mod___Mixed_7c_branch3x3dbl_2_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_2_bn_running_var = self.L__mod___Mixed_7c_branch3x3dbl_2_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch3x3dbl_2_bn_weight = self.L__mod___Mixed_7c_branch3x3dbl_2_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch3x3dbl_2_bn_bias = self.L__mod___Mixed_7c_branch3x3dbl_2_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_469 = torch.nn.functional.batch_norm(x_468, l__mod___mixed_7c_branch3x3dbl_2_bn_running_mean, l__mod___mixed_7c_branch3x3dbl_2_bn_running_var, l__mod___mixed_7c_branch3x3dbl_2_bn_weight, l__mod___mixed_7c_branch3x3dbl_2_bn_bias, True, 0.1, 0.001);  x_468 = l__mod___mixed_7c_branch3x3dbl_2_bn_running_mean = l__mod___mixed_7c_branch3x3dbl_2_bn_running_var = l__mod___mixed_7c_branch3x3dbl_2_bn_weight = l__mod___mixed_7c_branch3x3dbl_2_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_470 = self.L__mod___Mixed_7c_branch3x3dbl_2_bn_drop(x_469);  x_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch3x3dbl_16 = self.L__mod___Mixed_7c_branch3x3dbl_2_bn_act(x_470);  x_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_473 = self.L__mod___Mixed_7c_branch3x3dbl_3a_conv(branch3x3dbl_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch3x3dbl_3a_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch3x3dbl_3a_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__91 = l__mod___mixed_7c_branch3x3dbl_3a_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch3x3dbl_3a_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_3a_bn_running_mean = self.L__mod___Mixed_7c_branch3x3dbl_3a_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_3a_bn_running_var = self.L__mod___Mixed_7c_branch3x3dbl_3a_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch3x3dbl_3a_bn_weight = self.L__mod___Mixed_7c_branch3x3dbl_3a_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch3x3dbl_3a_bn_bias = self.L__mod___Mixed_7c_branch3x3dbl_3a_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_474 = torch.nn.functional.batch_norm(x_473, l__mod___mixed_7c_branch3x3dbl_3a_bn_running_mean, l__mod___mixed_7c_branch3x3dbl_3a_bn_running_var, l__mod___mixed_7c_branch3x3dbl_3a_bn_weight, l__mod___mixed_7c_branch3x3dbl_3a_bn_bias, True, 0.1, 0.001);  x_473 = l__mod___mixed_7c_branch3x3dbl_3a_bn_running_mean = l__mod___mixed_7c_branch3x3dbl_3a_bn_running_var = l__mod___mixed_7c_branch3x3dbl_3a_bn_weight = l__mod___mixed_7c_branch3x3dbl_3a_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_475 = self.L__mod___Mixed_7c_branch3x3dbl_3a_bn_drop(x_474);  x_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_477 = self.L__mod___Mixed_7c_branch3x3dbl_3a_bn_act(x_475);  x_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_478 = self.L__mod___Mixed_7c_branch3x3dbl_3b_conv(branch3x3dbl_16);  branch3x3dbl_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch3x3dbl_3b_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch3x3dbl_3b_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__92 = l__mod___mixed_7c_branch3x3dbl_3b_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch3x3dbl_3b_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_3b_bn_running_mean = self.L__mod___Mixed_7c_branch3x3dbl_3b_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch3x3dbl_3b_bn_running_var = self.L__mod___Mixed_7c_branch3x3dbl_3b_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch3x3dbl_3b_bn_weight = self.L__mod___Mixed_7c_branch3x3dbl_3b_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch3x3dbl_3b_bn_bias = self.L__mod___Mixed_7c_branch3x3dbl_3b_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_479 = torch.nn.functional.batch_norm(x_478, l__mod___mixed_7c_branch3x3dbl_3b_bn_running_mean, l__mod___mixed_7c_branch3x3dbl_3b_bn_running_var, l__mod___mixed_7c_branch3x3dbl_3b_bn_weight, l__mod___mixed_7c_branch3x3dbl_3b_bn_bias, True, 0.1, 0.001);  x_478 = l__mod___mixed_7c_branch3x3dbl_3b_bn_running_mean = l__mod___mixed_7c_branch3x3dbl_3b_bn_running_var = l__mod___mixed_7c_branch3x3dbl_3b_bn_weight = l__mod___mixed_7c_branch3x3dbl_3b_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_480 = self.L__mod___Mixed_7c_branch3x3dbl_3b_bn_drop(x_479);  x_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_482 = self.L__mod___Mixed_7c_branch3x3dbl_3b_bn_act(x_480);  x_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:196, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
    branch3x3dbl_17 = torch.cat([x_477, x_482], 1);  x_477 = x_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:198, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool_18 = torch._C._nn.avg_pool2d(x_442, kernel_size = 3, stride = 1, padding = 1);  x_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    x_483 = self.L__mod___Mixed_7c_branch_pool_conv(branch_pool_18);  branch_pool_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:97, code: if self.num_batches_tracked is not None:  # type: ignore[has-type]
    l__mod___mixed_7c_branch_pool_bn_num_batches_tracked = self.L__mod___Mixed_7c_branch_pool_bn_num_batches_tracked
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add__93 = l__mod___mixed_7c_branch_pool_bn_num_batches_tracked.add_(1);  l__mod___mixed_7c_branch_pool_bn_num_batches_tracked = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:121, code: self.running_mean if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch_pool_bn_running_mean = self.L__mod___Mixed_7c_branch_pool_bn_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___mixed_7c_branch_pool_bn_running_var = self.L__mod___Mixed_7c_branch_pool_bn_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___mixed_7c_branch_pool_bn_weight = self.L__mod___Mixed_7c_branch_pool_bn_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___mixed_7c_branch_pool_bn_bias = self.L__mod___Mixed_7c_branch_pool_bn_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_484 = torch.nn.functional.batch_norm(x_483, l__mod___mixed_7c_branch_pool_bn_running_mean, l__mod___mixed_7c_branch_pool_bn_running_var, l__mod___mixed_7c_branch_pool_bn_weight, l__mod___mixed_7c_branch_pool_bn_bias, True, 0.1, 0.001);  x_483 = l__mod___mixed_7c_branch_pool_bn_running_mean = l__mod___mixed_7c_branch_pool_bn_running_var = l__mod___mixed_7c_branch_pool_bn_weight = l__mod___mixed_7c_branch_pool_bn_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_485 = self.L__mod___Mixed_7c_branch_pool_bn_drop(x_484);  x_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    branch_pool_19 = self.L__mod___Mixed_7c_branch_pool_bn_act(x_485);  x_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:206, code: return torch.cat(outputs, 1)
    x_490 = torch.cat([branch1x1_8, branch3x3_6, branch3x3dbl_17, branch_pool_19], 1);  branch1x1_8 = branch3x3_6 = branch3x3dbl_17 = branch_pool_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_491 = self.L__mod___global_pool_pool(x_490);  x_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_493 = self.L__mod___global_pool_flatten(x_491);  x_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:376, code: x = self.head_drop(x)
    x_494 = self.L__mod___head_drop(x_493);  x_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:377, code: x = self.fc(x)
    pred = self.L__mod___fc(x_494);  x_494 = None
    return (pred,)
    