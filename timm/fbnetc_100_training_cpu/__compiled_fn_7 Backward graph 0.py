from __future__ import annotations



def forward(self, tangents_1: "f32[]"):
    # File: /workspace/youkaichao/code/pytorch/benchmarks/dynamo/timm_models.py:325, code: return reduce_to_scalar_loss(pred) / 1000.0
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 1000.0);  tangents_1 = None
    return [div_1]
    