from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:155, code: x = self.conv1(x)
    l__mod___conv1_0 = self.L__mod___conv1_0(l_inputs_0_);  l_inputs_0_ = None
    l__mod___conv1_1 = self.L__mod___conv1_1(l__mod___conv1_0);  l__mod___conv1_0 = None
    x = self.L__mod___conv1_2(l__mod___conv1_1);  l__mod___conv1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:156, code: x = self.maxpool(x)
    x_1 = self.L__mod___maxpool(x);  x = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    getattr_l__mod___stage2___0___branch1_0 = self.getattr_L__mod___stage2___0___branch1_0(x_1)
    getattr_l__mod___stage2___0___branch1_1 = self.getattr_L__mod___stage2___0___branch1_1(getattr_l__mod___stage2___0___branch1_0);  getattr_l__mod___stage2___0___branch1_0 = None
    getattr_l__mod___stage2___0___branch1_2 = self.getattr_L__mod___stage2___0___branch1_2(getattr_l__mod___stage2___0___branch1_1);  getattr_l__mod___stage2___0___branch1_1 = None
    getattr_l__mod___stage2___0___branch1_3 = self.getattr_L__mod___stage2___0___branch1_3(getattr_l__mod___stage2___0___branch1_2);  getattr_l__mod___stage2___0___branch1_2 = None
    getattr_l__mod___stage2___0___branch1_4 = self.getattr_L__mod___stage2___0___branch1_4(getattr_l__mod___stage2___0___branch1_3);  getattr_l__mod___stage2___0___branch1_3 = None
    getattr_l__mod___stage2___0___branch2_0 = self.getattr_L__mod___stage2___0___branch2_0(x_1);  x_1 = None
    getattr_l__mod___stage2___0___branch2_1 = self.getattr_L__mod___stage2___0___branch2_1(getattr_l__mod___stage2___0___branch2_0);  getattr_l__mod___stage2___0___branch2_0 = None
    getattr_l__mod___stage2___0___branch2_2 = self.getattr_L__mod___stage2___0___branch2_2(getattr_l__mod___stage2___0___branch2_1);  getattr_l__mod___stage2___0___branch2_1 = None
    getattr_l__mod___stage2___0___branch2_3 = self.getattr_L__mod___stage2___0___branch2_3(getattr_l__mod___stage2___0___branch2_2);  getattr_l__mod___stage2___0___branch2_2 = None
    getattr_l__mod___stage2___0___branch2_4 = self.getattr_L__mod___stage2___0___branch2_4(getattr_l__mod___stage2___0___branch2_3);  getattr_l__mod___stage2___0___branch2_3 = None
    getattr_l__mod___stage2___0___branch2_5 = self.getattr_L__mod___stage2___0___branch2_5(getattr_l__mod___stage2___0___branch2_4);  getattr_l__mod___stage2___0___branch2_4 = None
    getattr_l__mod___stage2___0___branch2_6 = self.getattr_L__mod___stage2___0___branch2_6(getattr_l__mod___stage2___0___branch2_5);  getattr_l__mod___stage2___0___branch2_5 = None
    getattr_l__mod___stage2___0___branch2_7 = self.getattr_L__mod___stage2___0___branch2_7(getattr_l__mod___stage2___0___branch2_6);  getattr_l__mod___stage2___0___branch2_6 = None
    out = torch.cat((getattr_l__mod___stage2___0___branch1_4, getattr_l__mod___stage2___0___branch2_7), dim = 1);  getattr_l__mod___stage2___0___branch1_4 = getattr_l__mod___stage2___0___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_2 = out.view(4, 2, 58, 28, 28);  out = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose = torch.transpose(x_2, 1, 2);  x_2 = None
    x_3 = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_1 = x_3.view(4, 116, 28, 28);  x_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk = out_1.chunk(2, dim = 1);  out_1 = None
    x1 = chunk[0]
    x2 = chunk[1];  chunk = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage2___1___branch2_0 = self.getattr_L__mod___stage2___1___branch2_0(x2);  x2 = None
    getattr_l__mod___stage2___1___branch2_1 = self.getattr_L__mod___stage2___1___branch2_1(getattr_l__mod___stage2___1___branch2_0);  getattr_l__mod___stage2___1___branch2_0 = None
    getattr_l__mod___stage2___1___branch2_2 = self.getattr_L__mod___stage2___1___branch2_2(getattr_l__mod___stage2___1___branch2_1);  getattr_l__mod___stage2___1___branch2_1 = None
    getattr_l__mod___stage2___1___branch2_3 = self.getattr_L__mod___stage2___1___branch2_3(getattr_l__mod___stage2___1___branch2_2);  getattr_l__mod___stage2___1___branch2_2 = None
    getattr_l__mod___stage2___1___branch2_4 = self.getattr_L__mod___stage2___1___branch2_4(getattr_l__mod___stage2___1___branch2_3);  getattr_l__mod___stage2___1___branch2_3 = None
    getattr_l__mod___stage2___1___branch2_5 = self.getattr_L__mod___stage2___1___branch2_5(getattr_l__mod___stage2___1___branch2_4);  getattr_l__mod___stage2___1___branch2_4 = None
    getattr_l__mod___stage2___1___branch2_6 = self.getattr_L__mod___stage2___1___branch2_6(getattr_l__mod___stage2___1___branch2_5);  getattr_l__mod___stage2___1___branch2_5 = None
    getattr_l__mod___stage2___1___branch2_7 = self.getattr_L__mod___stage2___1___branch2_7(getattr_l__mod___stage2___1___branch2_6);  getattr_l__mod___stage2___1___branch2_6 = None
    out_2 = torch.cat((x1, getattr_l__mod___stage2___1___branch2_7), dim = 1);  x1 = getattr_l__mod___stage2___1___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_5 = out_2.view(4, 2, 58, 28, 28);  out_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_1 = torch.transpose(x_5, 1, 2);  x_5 = None
    x_6 = transpose_1.contiguous();  transpose_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_3 = x_6.view(4, 116, 28, 28);  x_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_1 = out_3.chunk(2, dim = 1);  out_3 = None
    x1_1 = chunk_1[0]
    x2_1 = chunk_1[1];  chunk_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage2___2___branch2_0 = self.getattr_L__mod___stage2___2___branch2_0(x2_1);  x2_1 = None
    getattr_l__mod___stage2___2___branch2_1 = self.getattr_L__mod___stage2___2___branch2_1(getattr_l__mod___stage2___2___branch2_0);  getattr_l__mod___stage2___2___branch2_0 = None
    getattr_l__mod___stage2___2___branch2_2 = self.getattr_L__mod___stage2___2___branch2_2(getattr_l__mod___stage2___2___branch2_1);  getattr_l__mod___stage2___2___branch2_1 = None
    getattr_l__mod___stage2___2___branch2_3 = self.getattr_L__mod___stage2___2___branch2_3(getattr_l__mod___stage2___2___branch2_2);  getattr_l__mod___stage2___2___branch2_2 = None
    getattr_l__mod___stage2___2___branch2_4 = self.getattr_L__mod___stage2___2___branch2_4(getattr_l__mod___stage2___2___branch2_3);  getattr_l__mod___stage2___2___branch2_3 = None
    getattr_l__mod___stage2___2___branch2_5 = self.getattr_L__mod___stage2___2___branch2_5(getattr_l__mod___stage2___2___branch2_4);  getattr_l__mod___stage2___2___branch2_4 = None
    getattr_l__mod___stage2___2___branch2_6 = self.getattr_L__mod___stage2___2___branch2_6(getattr_l__mod___stage2___2___branch2_5);  getattr_l__mod___stage2___2___branch2_5 = None
    getattr_l__mod___stage2___2___branch2_7 = self.getattr_L__mod___stage2___2___branch2_7(getattr_l__mod___stage2___2___branch2_6);  getattr_l__mod___stage2___2___branch2_6 = None
    out_4 = torch.cat((x1_1, getattr_l__mod___stage2___2___branch2_7), dim = 1);  x1_1 = getattr_l__mod___stage2___2___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_8 = out_4.view(4, 2, 58, 28, 28);  out_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_2 = torch.transpose(x_8, 1, 2);  x_8 = None
    x_9 = transpose_2.contiguous();  transpose_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_5 = x_9.view(4, 116, 28, 28);  x_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_2 = out_5.chunk(2, dim = 1);  out_5 = None
    x1_2 = chunk_2[0]
    x2_2 = chunk_2[1];  chunk_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage2___3___branch2_0 = self.getattr_L__mod___stage2___3___branch2_0(x2_2);  x2_2 = None
    getattr_l__mod___stage2___3___branch2_1 = self.getattr_L__mod___stage2___3___branch2_1(getattr_l__mod___stage2___3___branch2_0);  getattr_l__mod___stage2___3___branch2_0 = None
    getattr_l__mod___stage2___3___branch2_2 = self.getattr_L__mod___stage2___3___branch2_2(getattr_l__mod___stage2___3___branch2_1);  getattr_l__mod___stage2___3___branch2_1 = None
    getattr_l__mod___stage2___3___branch2_3 = self.getattr_L__mod___stage2___3___branch2_3(getattr_l__mod___stage2___3___branch2_2);  getattr_l__mod___stage2___3___branch2_2 = None
    getattr_l__mod___stage2___3___branch2_4 = self.getattr_L__mod___stage2___3___branch2_4(getattr_l__mod___stage2___3___branch2_3);  getattr_l__mod___stage2___3___branch2_3 = None
    getattr_l__mod___stage2___3___branch2_5 = self.getattr_L__mod___stage2___3___branch2_5(getattr_l__mod___stage2___3___branch2_4);  getattr_l__mod___stage2___3___branch2_4 = None
    getattr_l__mod___stage2___3___branch2_6 = self.getattr_L__mod___stage2___3___branch2_6(getattr_l__mod___stage2___3___branch2_5);  getattr_l__mod___stage2___3___branch2_5 = None
    getattr_l__mod___stage2___3___branch2_7 = self.getattr_L__mod___stage2___3___branch2_7(getattr_l__mod___stage2___3___branch2_6);  getattr_l__mod___stage2___3___branch2_6 = None
    out_6 = torch.cat((x1_2, getattr_l__mod___stage2___3___branch2_7), dim = 1);  x1_2 = getattr_l__mod___stage2___3___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_11 = out_6.view(4, 2, 58, 28, 28);  out_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_3 = torch.transpose(x_11, 1, 2);  x_11 = None
    x_12 = transpose_3.contiguous();  transpose_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    x_14 = x_12.view(4, 116, 28, 28);  x_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    getattr_l__mod___stage3___0___branch1_0 = self.getattr_L__mod___stage3___0___branch1_0(x_14)
    getattr_l__mod___stage3___0___branch1_1 = self.getattr_L__mod___stage3___0___branch1_1(getattr_l__mod___stage3___0___branch1_0);  getattr_l__mod___stage3___0___branch1_0 = None
    getattr_l__mod___stage3___0___branch1_2 = self.getattr_L__mod___stage3___0___branch1_2(getattr_l__mod___stage3___0___branch1_1);  getattr_l__mod___stage3___0___branch1_1 = None
    getattr_l__mod___stage3___0___branch1_3 = self.getattr_L__mod___stage3___0___branch1_3(getattr_l__mod___stage3___0___branch1_2);  getattr_l__mod___stage3___0___branch1_2 = None
    getattr_l__mod___stage3___0___branch1_4 = self.getattr_L__mod___stage3___0___branch1_4(getattr_l__mod___stage3___0___branch1_3);  getattr_l__mod___stage3___0___branch1_3 = None
    getattr_l__mod___stage3___0___branch2_0 = self.getattr_L__mod___stage3___0___branch2_0(x_14);  x_14 = None
    getattr_l__mod___stage3___0___branch2_1 = self.getattr_L__mod___stage3___0___branch2_1(getattr_l__mod___stage3___0___branch2_0);  getattr_l__mod___stage3___0___branch2_0 = None
    getattr_l__mod___stage3___0___branch2_2 = self.getattr_L__mod___stage3___0___branch2_2(getattr_l__mod___stage3___0___branch2_1);  getattr_l__mod___stage3___0___branch2_1 = None
    getattr_l__mod___stage3___0___branch2_3 = self.getattr_L__mod___stage3___0___branch2_3(getattr_l__mod___stage3___0___branch2_2);  getattr_l__mod___stage3___0___branch2_2 = None
    getattr_l__mod___stage3___0___branch2_4 = self.getattr_L__mod___stage3___0___branch2_4(getattr_l__mod___stage3___0___branch2_3);  getattr_l__mod___stage3___0___branch2_3 = None
    getattr_l__mod___stage3___0___branch2_5 = self.getattr_L__mod___stage3___0___branch2_5(getattr_l__mod___stage3___0___branch2_4);  getattr_l__mod___stage3___0___branch2_4 = None
    getattr_l__mod___stage3___0___branch2_6 = self.getattr_L__mod___stage3___0___branch2_6(getattr_l__mod___stage3___0___branch2_5);  getattr_l__mod___stage3___0___branch2_5 = None
    getattr_l__mod___stage3___0___branch2_7 = self.getattr_L__mod___stage3___0___branch2_7(getattr_l__mod___stage3___0___branch2_6);  getattr_l__mod___stage3___0___branch2_6 = None
    out_8 = torch.cat((getattr_l__mod___stage3___0___branch1_4, getattr_l__mod___stage3___0___branch2_7), dim = 1);  getattr_l__mod___stage3___0___branch1_4 = getattr_l__mod___stage3___0___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_15 = out_8.view(4, 2, 116, 14, 14);  out_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_4 = torch.transpose(x_15, 1, 2);  x_15 = None
    x_16 = transpose_4.contiguous();  transpose_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_9 = x_16.view(4, 232, 14, 14);  x_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_3 = out_9.chunk(2, dim = 1);  out_9 = None
    x1_3 = chunk_3[0]
    x2_3 = chunk_3[1];  chunk_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage3___1___branch2_0 = self.getattr_L__mod___stage3___1___branch2_0(x2_3);  x2_3 = None
    getattr_l__mod___stage3___1___branch2_1 = self.getattr_L__mod___stage3___1___branch2_1(getattr_l__mod___stage3___1___branch2_0);  getattr_l__mod___stage3___1___branch2_0 = None
    getattr_l__mod___stage3___1___branch2_2 = self.getattr_L__mod___stage3___1___branch2_2(getattr_l__mod___stage3___1___branch2_1);  getattr_l__mod___stage3___1___branch2_1 = None
    getattr_l__mod___stage3___1___branch2_3 = self.getattr_L__mod___stage3___1___branch2_3(getattr_l__mod___stage3___1___branch2_2);  getattr_l__mod___stage3___1___branch2_2 = None
    getattr_l__mod___stage3___1___branch2_4 = self.getattr_L__mod___stage3___1___branch2_4(getattr_l__mod___stage3___1___branch2_3);  getattr_l__mod___stage3___1___branch2_3 = None
    getattr_l__mod___stage3___1___branch2_5 = self.getattr_L__mod___stage3___1___branch2_5(getattr_l__mod___stage3___1___branch2_4);  getattr_l__mod___stage3___1___branch2_4 = None
    getattr_l__mod___stage3___1___branch2_6 = self.getattr_L__mod___stage3___1___branch2_6(getattr_l__mod___stage3___1___branch2_5);  getattr_l__mod___stage3___1___branch2_5 = None
    getattr_l__mod___stage3___1___branch2_7 = self.getattr_L__mod___stage3___1___branch2_7(getattr_l__mod___stage3___1___branch2_6);  getattr_l__mod___stage3___1___branch2_6 = None
    out_10 = torch.cat((x1_3, getattr_l__mod___stage3___1___branch2_7), dim = 1);  x1_3 = getattr_l__mod___stage3___1___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_18 = out_10.view(4, 2, 116, 14, 14);  out_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_5 = torch.transpose(x_18, 1, 2);  x_18 = None
    x_19 = transpose_5.contiguous();  transpose_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_11 = x_19.view(4, 232, 14, 14);  x_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_4 = out_11.chunk(2, dim = 1);  out_11 = None
    x1_4 = chunk_4[0]
    x2_4 = chunk_4[1];  chunk_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage3___2___branch2_0 = self.getattr_L__mod___stage3___2___branch2_0(x2_4);  x2_4 = None
    getattr_l__mod___stage3___2___branch2_1 = self.getattr_L__mod___stage3___2___branch2_1(getattr_l__mod___stage3___2___branch2_0);  getattr_l__mod___stage3___2___branch2_0 = None
    getattr_l__mod___stage3___2___branch2_2 = self.getattr_L__mod___stage3___2___branch2_2(getattr_l__mod___stage3___2___branch2_1);  getattr_l__mod___stage3___2___branch2_1 = None
    getattr_l__mod___stage3___2___branch2_3 = self.getattr_L__mod___stage3___2___branch2_3(getattr_l__mod___stage3___2___branch2_2);  getattr_l__mod___stage3___2___branch2_2 = None
    getattr_l__mod___stage3___2___branch2_4 = self.getattr_L__mod___stage3___2___branch2_4(getattr_l__mod___stage3___2___branch2_3);  getattr_l__mod___stage3___2___branch2_3 = None
    getattr_l__mod___stage3___2___branch2_5 = self.getattr_L__mod___stage3___2___branch2_5(getattr_l__mod___stage3___2___branch2_4);  getattr_l__mod___stage3___2___branch2_4 = None
    getattr_l__mod___stage3___2___branch2_6 = self.getattr_L__mod___stage3___2___branch2_6(getattr_l__mod___stage3___2___branch2_5);  getattr_l__mod___stage3___2___branch2_5 = None
    getattr_l__mod___stage3___2___branch2_7 = self.getattr_L__mod___stage3___2___branch2_7(getattr_l__mod___stage3___2___branch2_6);  getattr_l__mod___stage3___2___branch2_6 = None
    out_12 = torch.cat((x1_4, getattr_l__mod___stage3___2___branch2_7), dim = 1);  x1_4 = getattr_l__mod___stage3___2___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_21 = out_12.view(4, 2, 116, 14, 14);  out_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_6 = torch.transpose(x_21, 1, 2);  x_21 = None
    x_22 = transpose_6.contiguous();  transpose_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_13 = x_22.view(4, 232, 14, 14);  x_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_5 = out_13.chunk(2, dim = 1);  out_13 = None
    x1_5 = chunk_5[0]
    x2_5 = chunk_5[1];  chunk_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage3___3___branch2_0 = self.getattr_L__mod___stage3___3___branch2_0(x2_5);  x2_5 = None
    getattr_l__mod___stage3___3___branch2_1 = self.getattr_L__mod___stage3___3___branch2_1(getattr_l__mod___stage3___3___branch2_0);  getattr_l__mod___stage3___3___branch2_0 = None
    getattr_l__mod___stage3___3___branch2_2 = self.getattr_L__mod___stage3___3___branch2_2(getattr_l__mod___stage3___3___branch2_1);  getattr_l__mod___stage3___3___branch2_1 = None
    getattr_l__mod___stage3___3___branch2_3 = self.getattr_L__mod___stage3___3___branch2_3(getattr_l__mod___stage3___3___branch2_2);  getattr_l__mod___stage3___3___branch2_2 = None
    getattr_l__mod___stage3___3___branch2_4 = self.getattr_L__mod___stage3___3___branch2_4(getattr_l__mod___stage3___3___branch2_3);  getattr_l__mod___stage3___3___branch2_3 = None
    getattr_l__mod___stage3___3___branch2_5 = self.getattr_L__mod___stage3___3___branch2_5(getattr_l__mod___stage3___3___branch2_4);  getattr_l__mod___stage3___3___branch2_4 = None
    getattr_l__mod___stage3___3___branch2_6 = self.getattr_L__mod___stage3___3___branch2_6(getattr_l__mod___stage3___3___branch2_5);  getattr_l__mod___stage3___3___branch2_5 = None
    getattr_l__mod___stage3___3___branch2_7 = self.getattr_L__mod___stage3___3___branch2_7(getattr_l__mod___stage3___3___branch2_6);  getattr_l__mod___stage3___3___branch2_6 = None
    out_14 = torch.cat((x1_5, getattr_l__mod___stage3___3___branch2_7), dim = 1);  x1_5 = getattr_l__mod___stage3___3___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_24 = out_14.view(4, 2, 116, 14, 14);  out_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_7 = torch.transpose(x_24, 1, 2);  x_24 = None
    x_25 = transpose_7.contiguous();  transpose_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_15 = x_25.view(4, 232, 14, 14);  x_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_6 = out_15.chunk(2, dim = 1);  out_15 = None
    x1_6 = chunk_6[0]
    x2_6 = chunk_6[1];  chunk_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage3___4___branch2_0 = self.getattr_L__mod___stage3___4___branch2_0(x2_6);  x2_6 = None
    getattr_l__mod___stage3___4___branch2_1 = self.getattr_L__mod___stage3___4___branch2_1(getattr_l__mod___stage3___4___branch2_0);  getattr_l__mod___stage3___4___branch2_0 = None
    getattr_l__mod___stage3___4___branch2_2 = self.getattr_L__mod___stage3___4___branch2_2(getattr_l__mod___stage3___4___branch2_1);  getattr_l__mod___stage3___4___branch2_1 = None
    getattr_l__mod___stage3___4___branch2_3 = self.getattr_L__mod___stage3___4___branch2_3(getattr_l__mod___stage3___4___branch2_2);  getattr_l__mod___stage3___4___branch2_2 = None
    getattr_l__mod___stage3___4___branch2_4 = self.getattr_L__mod___stage3___4___branch2_4(getattr_l__mod___stage3___4___branch2_3);  getattr_l__mod___stage3___4___branch2_3 = None
    getattr_l__mod___stage3___4___branch2_5 = self.getattr_L__mod___stage3___4___branch2_5(getattr_l__mod___stage3___4___branch2_4);  getattr_l__mod___stage3___4___branch2_4 = None
    getattr_l__mod___stage3___4___branch2_6 = self.getattr_L__mod___stage3___4___branch2_6(getattr_l__mod___stage3___4___branch2_5);  getattr_l__mod___stage3___4___branch2_5 = None
    getattr_l__mod___stage3___4___branch2_7 = self.getattr_L__mod___stage3___4___branch2_7(getattr_l__mod___stage3___4___branch2_6);  getattr_l__mod___stage3___4___branch2_6 = None
    out_16 = torch.cat((x1_6, getattr_l__mod___stage3___4___branch2_7), dim = 1);  x1_6 = getattr_l__mod___stage3___4___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_27 = out_16.view(4, 2, 116, 14, 14);  out_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_8 = torch.transpose(x_27, 1, 2);  x_27 = None
    x_28 = transpose_8.contiguous();  transpose_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_17 = x_28.view(4, 232, 14, 14);  x_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_7 = out_17.chunk(2, dim = 1);  out_17 = None
    x1_7 = chunk_7[0]
    x2_7 = chunk_7[1];  chunk_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage3___5___branch2_0 = self.getattr_L__mod___stage3___5___branch2_0(x2_7);  x2_7 = None
    getattr_l__mod___stage3___5___branch2_1 = self.getattr_L__mod___stage3___5___branch2_1(getattr_l__mod___stage3___5___branch2_0);  getattr_l__mod___stage3___5___branch2_0 = None
    getattr_l__mod___stage3___5___branch2_2 = self.getattr_L__mod___stage3___5___branch2_2(getattr_l__mod___stage3___5___branch2_1);  getattr_l__mod___stage3___5___branch2_1 = None
    getattr_l__mod___stage3___5___branch2_3 = self.getattr_L__mod___stage3___5___branch2_3(getattr_l__mod___stage3___5___branch2_2);  getattr_l__mod___stage3___5___branch2_2 = None
    getattr_l__mod___stage3___5___branch2_4 = self.getattr_L__mod___stage3___5___branch2_4(getattr_l__mod___stage3___5___branch2_3);  getattr_l__mod___stage3___5___branch2_3 = None
    getattr_l__mod___stage3___5___branch2_5 = self.getattr_L__mod___stage3___5___branch2_5(getattr_l__mod___stage3___5___branch2_4);  getattr_l__mod___stage3___5___branch2_4 = None
    getattr_l__mod___stage3___5___branch2_6 = self.getattr_L__mod___stage3___5___branch2_6(getattr_l__mod___stage3___5___branch2_5);  getattr_l__mod___stage3___5___branch2_5 = None
    getattr_l__mod___stage3___5___branch2_7 = self.getattr_L__mod___stage3___5___branch2_7(getattr_l__mod___stage3___5___branch2_6);  getattr_l__mod___stage3___5___branch2_6 = None
    out_18 = torch.cat((x1_7, getattr_l__mod___stage3___5___branch2_7), dim = 1);  x1_7 = getattr_l__mod___stage3___5___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_30 = out_18.view(4, 2, 116, 14, 14);  out_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_9 = torch.transpose(x_30, 1, 2);  x_30 = None
    x_31 = transpose_9.contiguous();  transpose_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_19 = x_31.view(4, 232, 14, 14);  x_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_8 = out_19.chunk(2, dim = 1);  out_19 = None
    x1_8 = chunk_8[0]
    x2_8 = chunk_8[1];  chunk_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage3___6___branch2_0 = self.getattr_L__mod___stage3___6___branch2_0(x2_8);  x2_8 = None
    getattr_l__mod___stage3___6___branch2_1 = self.getattr_L__mod___stage3___6___branch2_1(getattr_l__mod___stage3___6___branch2_0);  getattr_l__mod___stage3___6___branch2_0 = None
    getattr_l__mod___stage3___6___branch2_2 = self.getattr_L__mod___stage3___6___branch2_2(getattr_l__mod___stage3___6___branch2_1);  getattr_l__mod___stage3___6___branch2_1 = None
    getattr_l__mod___stage3___6___branch2_3 = self.getattr_L__mod___stage3___6___branch2_3(getattr_l__mod___stage3___6___branch2_2);  getattr_l__mod___stage3___6___branch2_2 = None
    getattr_l__mod___stage3___6___branch2_4 = self.getattr_L__mod___stage3___6___branch2_4(getattr_l__mod___stage3___6___branch2_3);  getattr_l__mod___stage3___6___branch2_3 = None
    getattr_l__mod___stage3___6___branch2_5 = self.getattr_L__mod___stage3___6___branch2_5(getattr_l__mod___stage3___6___branch2_4);  getattr_l__mod___stage3___6___branch2_4 = None
    getattr_l__mod___stage3___6___branch2_6 = self.getattr_L__mod___stage3___6___branch2_6(getattr_l__mod___stage3___6___branch2_5);  getattr_l__mod___stage3___6___branch2_5 = None
    getattr_l__mod___stage3___6___branch2_7 = self.getattr_L__mod___stage3___6___branch2_7(getattr_l__mod___stage3___6___branch2_6);  getattr_l__mod___stage3___6___branch2_6 = None
    out_20 = torch.cat((x1_8, getattr_l__mod___stage3___6___branch2_7), dim = 1);  x1_8 = getattr_l__mod___stage3___6___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_33 = out_20.view(4, 2, 116, 14, 14);  out_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_10 = torch.transpose(x_33, 1, 2);  x_33 = None
    x_34 = transpose_10.contiguous();  transpose_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_21 = x_34.view(4, 232, 14, 14);  x_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_9 = out_21.chunk(2, dim = 1);  out_21 = None
    x1_9 = chunk_9[0]
    x2_9 = chunk_9[1];  chunk_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage3___7___branch2_0 = self.getattr_L__mod___stage3___7___branch2_0(x2_9);  x2_9 = None
    getattr_l__mod___stage3___7___branch2_1 = self.getattr_L__mod___stage3___7___branch2_1(getattr_l__mod___stage3___7___branch2_0);  getattr_l__mod___stage3___7___branch2_0 = None
    getattr_l__mod___stage3___7___branch2_2 = self.getattr_L__mod___stage3___7___branch2_2(getattr_l__mod___stage3___7___branch2_1);  getattr_l__mod___stage3___7___branch2_1 = None
    getattr_l__mod___stage3___7___branch2_3 = self.getattr_L__mod___stage3___7___branch2_3(getattr_l__mod___stage3___7___branch2_2);  getattr_l__mod___stage3___7___branch2_2 = None
    getattr_l__mod___stage3___7___branch2_4 = self.getattr_L__mod___stage3___7___branch2_4(getattr_l__mod___stage3___7___branch2_3);  getattr_l__mod___stage3___7___branch2_3 = None
    getattr_l__mod___stage3___7___branch2_5 = self.getattr_L__mod___stage3___7___branch2_5(getattr_l__mod___stage3___7___branch2_4);  getattr_l__mod___stage3___7___branch2_4 = None
    getattr_l__mod___stage3___7___branch2_6 = self.getattr_L__mod___stage3___7___branch2_6(getattr_l__mod___stage3___7___branch2_5);  getattr_l__mod___stage3___7___branch2_5 = None
    getattr_l__mod___stage3___7___branch2_7 = self.getattr_L__mod___stage3___7___branch2_7(getattr_l__mod___stage3___7___branch2_6);  getattr_l__mod___stage3___7___branch2_6 = None
    out_22 = torch.cat((x1_9, getattr_l__mod___stage3___7___branch2_7), dim = 1);  x1_9 = getattr_l__mod___stage3___7___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_36 = out_22.view(4, 2, 116, 14, 14);  out_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_11 = torch.transpose(x_36, 1, 2);  x_36 = None
    x_37 = transpose_11.contiguous();  transpose_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    x_39 = x_37.view(4, 232, 14, 14);  x_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:97, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    getattr_l__mod___stage4___0___branch1_0 = self.getattr_L__mod___stage4___0___branch1_0(x_39)
    getattr_l__mod___stage4___0___branch1_1 = self.getattr_L__mod___stage4___0___branch1_1(getattr_l__mod___stage4___0___branch1_0);  getattr_l__mod___stage4___0___branch1_0 = None
    getattr_l__mod___stage4___0___branch1_2 = self.getattr_L__mod___stage4___0___branch1_2(getattr_l__mod___stage4___0___branch1_1);  getattr_l__mod___stage4___0___branch1_1 = None
    getattr_l__mod___stage4___0___branch1_3 = self.getattr_L__mod___stage4___0___branch1_3(getattr_l__mod___stage4___0___branch1_2);  getattr_l__mod___stage4___0___branch1_2 = None
    getattr_l__mod___stage4___0___branch1_4 = self.getattr_L__mod___stage4___0___branch1_4(getattr_l__mod___stage4___0___branch1_3);  getattr_l__mod___stage4___0___branch1_3 = None
    getattr_l__mod___stage4___0___branch2_0 = self.getattr_L__mod___stage4___0___branch2_0(x_39);  x_39 = None
    getattr_l__mod___stage4___0___branch2_1 = self.getattr_L__mod___stage4___0___branch2_1(getattr_l__mod___stage4___0___branch2_0);  getattr_l__mod___stage4___0___branch2_0 = None
    getattr_l__mod___stage4___0___branch2_2 = self.getattr_L__mod___stage4___0___branch2_2(getattr_l__mod___stage4___0___branch2_1);  getattr_l__mod___stage4___0___branch2_1 = None
    getattr_l__mod___stage4___0___branch2_3 = self.getattr_L__mod___stage4___0___branch2_3(getattr_l__mod___stage4___0___branch2_2);  getattr_l__mod___stage4___0___branch2_2 = None
    getattr_l__mod___stage4___0___branch2_4 = self.getattr_L__mod___stage4___0___branch2_4(getattr_l__mod___stage4___0___branch2_3);  getattr_l__mod___stage4___0___branch2_3 = None
    getattr_l__mod___stage4___0___branch2_5 = self.getattr_L__mod___stage4___0___branch2_5(getattr_l__mod___stage4___0___branch2_4);  getattr_l__mod___stage4___0___branch2_4 = None
    getattr_l__mod___stage4___0___branch2_6 = self.getattr_L__mod___stage4___0___branch2_6(getattr_l__mod___stage4___0___branch2_5);  getattr_l__mod___stage4___0___branch2_5 = None
    getattr_l__mod___stage4___0___branch2_7 = self.getattr_L__mod___stage4___0___branch2_7(getattr_l__mod___stage4___0___branch2_6);  getattr_l__mod___stage4___0___branch2_6 = None
    out_24 = torch.cat((getattr_l__mod___stage4___0___branch1_4, getattr_l__mod___stage4___0___branch2_7), dim = 1);  getattr_l__mod___stage4___0___branch1_4 = getattr_l__mod___stage4___0___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_40 = out_24.view(4, 2, 232, 7, 7);  out_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_12 = torch.transpose(x_40, 1, 2);  x_40 = None
    x_41 = transpose_12.contiguous();  transpose_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_25 = x_41.view(4, 464, 7, 7);  x_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_10 = out_25.chunk(2, dim = 1);  out_25 = None
    x1_10 = chunk_10[0]
    x2_10 = chunk_10[1];  chunk_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage4___1___branch2_0 = self.getattr_L__mod___stage4___1___branch2_0(x2_10);  x2_10 = None
    getattr_l__mod___stage4___1___branch2_1 = self.getattr_L__mod___stage4___1___branch2_1(getattr_l__mod___stage4___1___branch2_0);  getattr_l__mod___stage4___1___branch2_0 = None
    getattr_l__mod___stage4___1___branch2_2 = self.getattr_L__mod___stage4___1___branch2_2(getattr_l__mod___stage4___1___branch2_1);  getattr_l__mod___stage4___1___branch2_1 = None
    getattr_l__mod___stage4___1___branch2_3 = self.getattr_L__mod___stage4___1___branch2_3(getattr_l__mod___stage4___1___branch2_2);  getattr_l__mod___stage4___1___branch2_2 = None
    getattr_l__mod___stage4___1___branch2_4 = self.getattr_L__mod___stage4___1___branch2_4(getattr_l__mod___stage4___1___branch2_3);  getattr_l__mod___stage4___1___branch2_3 = None
    getattr_l__mod___stage4___1___branch2_5 = self.getattr_L__mod___stage4___1___branch2_5(getattr_l__mod___stage4___1___branch2_4);  getattr_l__mod___stage4___1___branch2_4 = None
    getattr_l__mod___stage4___1___branch2_6 = self.getattr_L__mod___stage4___1___branch2_6(getattr_l__mod___stage4___1___branch2_5);  getattr_l__mod___stage4___1___branch2_5 = None
    getattr_l__mod___stage4___1___branch2_7 = self.getattr_L__mod___stage4___1___branch2_7(getattr_l__mod___stage4___1___branch2_6);  getattr_l__mod___stage4___1___branch2_6 = None
    out_26 = torch.cat((x1_10, getattr_l__mod___stage4___1___branch2_7), dim = 1);  x1_10 = getattr_l__mod___stage4___1___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_43 = out_26.view(4, 2, 232, 7, 7);  out_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_13 = torch.transpose(x_43, 1, 2);  x_43 = None
    x_44 = transpose_13.contiguous();  transpose_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_27 = x_44.view(4, 464, 7, 7);  x_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_11 = out_27.chunk(2, dim = 1);  out_27 = None
    x1_11 = chunk_11[0]
    x2_11 = chunk_11[1];  chunk_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage4___2___branch2_0 = self.getattr_L__mod___stage4___2___branch2_0(x2_11);  x2_11 = None
    getattr_l__mod___stage4___2___branch2_1 = self.getattr_L__mod___stage4___2___branch2_1(getattr_l__mod___stage4___2___branch2_0);  getattr_l__mod___stage4___2___branch2_0 = None
    getattr_l__mod___stage4___2___branch2_2 = self.getattr_L__mod___stage4___2___branch2_2(getattr_l__mod___stage4___2___branch2_1);  getattr_l__mod___stage4___2___branch2_1 = None
    getattr_l__mod___stage4___2___branch2_3 = self.getattr_L__mod___stage4___2___branch2_3(getattr_l__mod___stage4___2___branch2_2);  getattr_l__mod___stage4___2___branch2_2 = None
    getattr_l__mod___stage4___2___branch2_4 = self.getattr_L__mod___stage4___2___branch2_4(getattr_l__mod___stage4___2___branch2_3);  getattr_l__mod___stage4___2___branch2_3 = None
    getattr_l__mod___stage4___2___branch2_5 = self.getattr_L__mod___stage4___2___branch2_5(getattr_l__mod___stage4___2___branch2_4);  getattr_l__mod___stage4___2___branch2_4 = None
    getattr_l__mod___stage4___2___branch2_6 = self.getattr_L__mod___stage4___2___branch2_6(getattr_l__mod___stage4___2___branch2_5);  getattr_l__mod___stage4___2___branch2_5 = None
    getattr_l__mod___stage4___2___branch2_7 = self.getattr_L__mod___stage4___2___branch2_7(getattr_l__mod___stage4___2___branch2_6);  getattr_l__mod___stage4___2___branch2_6 = None
    out_28 = torch.cat((x1_11, getattr_l__mod___stage4___2___branch2_7), dim = 1);  x1_11 = getattr_l__mod___stage4___2___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_46 = out_28.view(4, 2, 232, 7, 7);  out_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_14 = torch.transpose(x_46, 1, 2);  x_46 = None
    x_47 = transpose_14.contiguous();  transpose_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    out_29 = x_47.view(4, 464, 7, 7);  x_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:94, code: x1, x2 = x.chunk(2, dim=1)
    chunk_12 = out_29.chunk(2, dim = 1);  out_29 = None
    x1_12 = chunk_12[0]
    x2_12 = chunk_12[1];  chunk_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:95, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
    getattr_l__mod___stage4___3___branch2_0 = self.getattr_L__mod___stage4___3___branch2_0(x2_12);  x2_12 = None
    getattr_l__mod___stage4___3___branch2_1 = self.getattr_L__mod___stage4___3___branch2_1(getattr_l__mod___stage4___3___branch2_0);  getattr_l__mod___stage4___3___branch2_0 = None
    getattr_l__mod___stage4___3___branch2_2 = self.getattr_L__mod___stage4___3___branch2_2(getattr_l__mod___stage4___3___branch2_1);  getattr_l__mod___stage4___3___branch2_1 = None
    getattr_l__mod___stage4___3___branch2_3 = self.getattr_L__mod___stage4___3___branch2_3(getattr_l__mod___stage4___3___branch2_2);  getattr_l__mod___stage4___3___branch2_2 = None
    getattr_l__mod___stage4___3___branch2_4 = self.getattr_L__mod___stage4___3___branch2_4(getattr_l__mod___stage4___3___branch2_3);  getattr_l__mod___stage4___3___branch2_3 = None
    getattr_l__mod___stage4___3___branch2_5 = self.getattr_L__mod___stage4___3___branch2_5(getattr_l__mod___stage4___3___branch2_4);  getattr_l__mod___stage4___3___branch2_4 = None
    getattr_l__mod___stage4___3___branch2_6 = self.getattr_L__mod___stage4___3___branch2_6(getattr_l__mod___stage4___3___branch2_5);  getattr_l__mod___stage4___3___branch2_5 = None
    getattr_l__mod___stage4___3___branch2_7 = self.getattr_L__mod___stage4___3___branch2_7(getattr_l__mod___stage4___3___branch2_6);  getattr_l__mod___stage4___3___branch2_6 = None
    out_30 = torch.cat((x1_12, getattr_l__mod___stage4___3___branch2_7), dim = 1);  x1_12 = getattr_l__mod___stage4___3___branch2_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:33, code: x = x.view(batchsize, groups, channels_per_group, height, width)
    x_49 = out_30.view(4, 2, 232, 7, 7);  out_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:35, code: x = torch.transpose(x, 1, 2).contiguous()
    transpose_15 = torch.transpose(x_49, 1, 2);  x_49 = None
    x_50 = transpose_15.contiguous();  transpose_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:38, code: x = x.view(batchsize, num_channels, height, width)
    x_52 = x_50.view(4, 464, 7, 7);  x_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:160, code: x = self.conv5(x)
    l__mod___conv5_0 = self.L__mod___conv5_0(x_52);  x_52 = None
    l__mod___conv5_1 = self.L__mod___conv5_1(l__mod___conv5_0);  l__mod___conv5_0 = None
    x_53 = self.L__mod___conv5_2(l__mod___conv5_1);  l__mod___conv5_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:161, code: x = x.mean([2, 3])  # globalpool
    x_54 = x_53.mean([2, 3]);  x_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/shufflenetv2.py:162, code: x = self.fc(x)
    x_55 = self.L__mod___fc(x_54);  x_54 = None
    return (x_55,)
    