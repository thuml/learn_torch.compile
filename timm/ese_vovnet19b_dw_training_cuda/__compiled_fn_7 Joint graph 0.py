from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[]"; tangents_1: "f32[]"; 

    primals_1, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/pytorch/benchmarks/dynamo/timm_models.py:325, code: return reduce_to_scalar_loss(pred) / 1000.0
    div: "f32[]" = torch.ops.aten.div.Tensor(primals_1, 1000.0);  primals_1 = None
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 1000.0);  tangents_1 = None
    return pytree.tree_unflatten([div, div_1], self._out_spec)
    