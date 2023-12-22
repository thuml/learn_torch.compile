from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:100, code: for ind, w in enumerate(self.encode_w):
    w = self.L__mod___encode_w_0
    w_1 = self.L__mod___encode_w_1
    w_2 = self.L__mod___encode_w_2
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    l__mod___encode_b_0 = self.L__mod___encode_b_0
    linear = torch._C._nn.linear(input = l_cloned_inputs_0_, weight = w, bias = l__mod___encode_b_0);  l_cloned_inputs_0_ = w = l__mod___encode_b_0 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    x = torch.nn.functional.selu(linear);  linear = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    l__mod___encode_b_1 = self.L__mod___encode_b_1
    linear_1 = torch._C._nn.linear(input = x, weight = w_1, bias = l__mod___encode_b_1);  x = w_1 = l__mod___encode_b_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    x_1 = torch.nn.functional.selu(linear_1);  linear_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:101, code: x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    l__mod___encode_b_2 = self.L__mod___encode_b_2
    linear_2 = torch._C._nn.linear(input = x_1, weight = w_2, bias = l__mod___encode_b_2);  x_1 = w_2 = l__mod___encode_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    x_2 = torch.nn.functional.selu(linear_2);  linear_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:103, code: x = self.drop(x)
    x_3 = self.L__mod___drop(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:115, code: for ind, w in enumerate(self.decode_w):
    w_3 = self.L__mod___decode_w_0
    w_4 = self.L__mod___decode_w_1
    w_5 = self.L__mod___decode_w_2
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    l__mod___decode_b_0 = self.L__mod___decode_b_0
    linear_3 = torch._C._nn.linear(input = x_3, weight = w_3, bias = l__mod___decode_b_0);  x_3 = w_3 = l__mod___decode_b_0 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    z = torch.nn.functional.selu(linear_3);  linear_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    l__mod___decode_b_1 = self.L__mod___decode_b_1
    linear_4 = torch._C._nn.linear(input = z, weight = w_4, bias = l__mod___decode_b_1);  z = w_4 = l__mod___decode_b_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    z_1 = torch.nn.functional.selu(linear_4);  linear_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:116, code: z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
    l__mod___decode_b_2 = self.L__mod___decode_b_2
    linear_5 = torch._C._nn.linear(input = z_1, weight = w_5, bias = l__mod___decode_b_2);  z_1 = w_5 = l__mod___decode_b_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:11, code: return F.selu(input)
    pred = torch.nn.functional.selu(linear_5);  linear_5 = None
    return (pred,)
    