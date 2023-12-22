from __future__ import annotations



def forward(self, primals_1: "f32[]"):
    # File: /workspace/youkaichao/code/pytorch/benchmarks/dynamo/timm_models.py:325, code: return reduce_to_scalar_loss(pred) / 1000.0
    div: "f32[]" = torch.ops.aten.div.Tensor(primals_1, 1000.0);  primals_1 = None
    return [div]
    