
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gmguyjzl35y52vobhacdogcxj4bzu2rol3fqph5morsk2pc3v4.py
# Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
# token_type_ids => full_default
triton_poi_fused_zeros_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ev/cevdhbe7skn5s3j64ktj5nkn3a3fspposdr4qkuxvh4evdrhxxuz.py
# Source Nodes: [embeddings, embeddings_1, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding]
# embeddings => add
# embeddings_1 => add_1
# inputs_embeds => embedding
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
triton_poi_fused_add_embedding_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 29056
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 29056), "index out of bounds: 0 <= tmp3 < 29056")
    tmp4 = tl.load(in_ptr1 + (x0 + (1024*tmp3)), None)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 + 512
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 512), "index out of bounds: 0 <= tmp10 < 512")
    tmp11 = tl.load(in_ptr4 + (x0 + (1024*tmp10)), None)
    tmp12 = tmp6 + tmp11
    tl.store(out_ptr0 + (x2), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewf2te3ya6eofzwylib5d4knpavnyusnb5osdusp2dqh43fuakg.py
# Source Nodes: [ln_outputs, mixed_query_layer], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# ln_outputs => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# mixed_query_layer => view
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 1024.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkeryg26pwftzyayut2no6nuddjxtwgsk5rdmo7mvlezy4724qp.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm75caxni3j4gl6azplcbi4gxyjr7yooy37qmy7sgd644u3hb5kd.py
# Source Nodes: [hidden_states], Original ATen: [aten.view]
# hidden_states => view_16
triton_poi_fused_view_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjssdoy65y3kyjxblfu6roszjrgw2j4vr7dsjzcoshqtkosf44j.py
# Source Nodes: [attention_output, hidden_states_2, ln_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# attention_output => add_5
# hidden_states_2 => view_18
# ln_output => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 1024.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csr5on5gptdxg35rko23fcfhxaqnespa3nxwxmslxkbinusqghny.py
# Source Nodes: [hidden_states_4, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_4 => view_20
# intermediate_output => add_8, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crx2gu3igjwr7aigethqhnqswsnbemwa7ko53qol4dcjy6bdqchy.py
# Source Nodes: [attention_output, hidden_states_6, ln_outputs_1, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# attention_output => add_5
# hidden_states_6 => add_9
# ln_outputs_1 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_4, var_mean_2
# mixed_query_layer_1 => view_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6kygtnf4ikgbacno5nnlqxh2mfpnnerkkfqqcusafhtztxqntn.py
# Source Nodes: [attention_output, attention_output_2, hidden_states_6, hidden_states_9, ln_output_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# attention_output => add_5
# attention_output_2 => add_13
# hidden_states_6 => add_9
# hidden_states_9 => view_40
# ln_output_1 => add_14, add_15, mul_10, mul_11, rsqrt_3, sub_6, var_mean_3
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 1024, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 1024.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 / tmp24
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdotyts2imlqfwxlcebdnehkyrd6pkhphqxuuc5hrn7hzpkdyu3w.py
# Source Nodes: [attention_output, attention_output_2, hidden_states_13, hidden_states_6, ln_outputs_2, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# attention_output => add_5
# attention_output_2 => add_13
# hidden_states_13 => add_17
# hidden_states_6 => add_9
# ln_outputs_2 => add_18, add_19, mul_15, mul_16, rsqrt_4, sub_7, var_mean_4
# mixed_query_layer_2 => view_44
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 1024.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-12
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curpboy5q7n5dha4nih5zvod6hi3qrqr5h7rqsikynrku2ipashf.py
# Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
# start_logits_1 => clone_24
# start_loss => amax_24, exp_24, log, sub_74, sub_75, sum_25
triton_per_fused__log_softmax_clone_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (2*r0), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3zvucjscm7pl7jktzvylbouevy5l754mr2hg7wikgvonplcmse.py
# Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
# end_logits_1 => clone_25
# end_loss => amax_25, exp_25, log_1, sub_76, sub_77, sum_28
triton_per_fused__log_softmax_clone_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (1 + (2*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (1))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xj/cxjiwxdvg2le5wf5rsi2453pjy4yung6lye66tayrpgui6z55hrc.py
# Source Nodes: [add_73, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# add_73 => add_196
# end_loss => convert_element_type_1, div_49, ne_3, neg_1, sum_29, sum_30, where_3
# end_positions => clamp_max_1, clamp_min_1
# loss => div_50
# start_loss => convert_element_type, div_48, full_default_2, full_default_3, ne, neg, sum_26, sum_27, where_1
# start_positions => clamp_max, clamp_min
triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: '*fp32', 7: '*i1', 8: '*i64', 9: '*i1', 10: '*i64', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp9 = triton_helpers.maximum(tmp8, tmp2)
    tmp10 = triton_helpers.minimum(tmp9, tmp4)
    tmp11 = tmp10 != tmp4
    tmp12 = tl.where(tmp6, tmp5, tmp2)
    tmp13 = tmp12 + 512
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 512), "index out of bounds: 0 <= tmp15 < 512")
    tmp16 = tl.load(in_ptr2 + (tmp15), None, eviction_policy='evict_last')
    tmp17 = -tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tmp6.to(tl.int64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tl.where(tmp11, tmp10, tmp2)
    tmp24 = tmp23 + 512
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert((0 <= tmp26) & (tmp26 < 512), "index out of bounds: 0 <= tmp26 < 512")
    tmp27 = tl.load(in_ptr3 + (tmp26), None, eviction_policy='evict_last')
    tmp28 = -tmp27
    tmp29 = tl.where(tmp11, tmp28, tmp18)
    tmp30 = tmp11.to(tl.int64)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp22 + tmp32
    tmp34 = 2.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp23, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395 = args
    args.clear()
    assert_size_stride(primals_1, (29056, 1024), (1024, 1))
    assert_size_stride(primals_2, (2, 1024), (1024, 1))
    assert_size_stride(primals_3, (512, 1024), (1024, 1))
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_5, (1024, ), (1, ))
    assert_size_stride(primals_6, (1024, 1024), (1024, 1))
    assert_size_stride(primals_7, (1024, ), (1, ))
    assert_size_stride(primals_8, (1024, 1024), (1024, 1))
    assert_size_stride(primals_9, (1024, ), (1, ))
    assert_size_stride(primals_10, (1024, 1024), (1024, 1))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, 1024), (1024, 1))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_15, (1024, ), (1, ))
    assert_size_stride(primals_16, (4096, 1024), (1024, 1))
    assert_size_stride(primals_17, (4096, ), (1, ))
    assert_size_stride(primals_18, (1024, 4096), (4096, 1))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(primals_21, (1024, ), (1, ))
    assert_size_stride(primals_22, (1024, 1024), (1024, 1))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (1024, 1024), (1024, 1))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(primals_26, (1024, 1024), (1024, 1))
    assert_size_stride(primals_27, (1024, ), (1, ))
    assert_size_stride(primals_28, (1024, 1024), (1024, 1))
    assert_size_stride(primals_29, (1024, ), (1, ))
    assert_size_stride(primals_30, (1024, ), (1, ))
    assert_size_stride(primals_31, (1024, ), (1, ))
    assert_size_stride(primals_32, (4096, 1024), (1024, 1))
    assert_size_stride(primals_33, (4096, ), (1, ))
    assert_size_stride(primals_34, (1024, 4096), (4096, 1))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, ), (1, ))
    assert_size_stride(primals_37, (1024, ), (1, ))
    assert_size_stride(primals_38, (1024, 1024), (1024, 1))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_40, (1024, 1024), (1024, 1))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_42, (1024, 1024), (1024, 1))
    assert_size_stride(primals_43, (1024, ), (1, ))
    assert_size_stride(primals_44, (1024, 1024), (1024, 1))
    assert_size_stride(primals_45, (1024, ), (1, ))
    assert_size_stride(primals_46, (1024, ), (1, ))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_48, (4096, 1024), (1024, 1))
    assert_size_stride(primals_49, (4096, ), (1, ))
    assert_size_stride(primals_50, (1024, 4096), (4096, 1))
    assert_size_stride(primals_51, (1024, ), (1, ))
    assert_size_stride(primals_52, (1024, ), (1, ))
    assert_size_stride(primals_53, (1024, ), (1, ))
    assert_size_stride(primals_54, (1024, 1024), (1024, 1))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_56, (1024, 1024), (1024, 1))
    assert_size_stride(primals_57, (1024, ), (1, ))
    assert_size_stride(primals_58, (1024, 1024), (1024, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_60, (1024, 1024), (1024, 1))
    assert_size_stride(primals_61, (1024, ), (1, ))
    assert_size_stride(primals_62, (1024, ), (1, ))
    assert_size_stride(primals_63, (1024, ), (1, ))
    assert_size_stride(primals_64, (4096, 1024), (1024, 1))
    assert_size_stride(primals_65, (4096, ), (1, ))
    assert_size_stride(primals_66, (1024, 4096), (4096, 1))
    assert_size_stride(primals_67, (1024, ), (1, ))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_69, (1024, ), (1, ))
    assert_size_stride(primals_70, (1024, 1024), (1024, 1))
    assert_size_stride(primals_71, (1024, ), (1, ))
    assert_size_stride(primals_72, (1024, 1024), (1024, 1))
    assert_size_stride(primals_73, (1024, ), (1, ))
    assert_size_stride(primals_74, (1024, 1024), (1024, 1))
    assert_size_stride(primals_75, (1024, ), (1, ))
    assert_size_stride(primals_76, (1024, 1024), (1024, 1))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_78, (1024, ), (1, ))
    assert_size_stride(primals_79, (1024, ), (1, ))
    assert_size_stride(primals_80, (4096, 1024), (1024, 1))
    assert_size_stride(primals_81, (4096, ), (1, ))
    assert_size_stride(primals_82, (1024, 4096), (4096, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (1024, ), (1, ))
    assert_size_stride(primals_86, (1024, 1024), (1024, 1))
    assert_size_stride(primals_87, (1024, ), (1, ))
    assert_size_stride(primals_88, (1024, 1024), (1024, 1))
    assert_size_stride(primals_89, (1024, ), (1, ))
    assert_size_stride(primals_90, (1024, 1024), (1024, 1))
    assert_size_stride(primals_91, (1024, ), (1, ))
    assert_size_stride(primals_92, (1024, 1024), (1024, 1))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (1024, ), (1, ))
    assert_size_stride(primals_95, (1024, ), (1, ))
    assert_size_stride(primals_96, (4096, 1024), (1024, 1))
    assert_size_stride(primals_97, (4096, ), (1, ))
    assert_size_stride(primals_98, (1024, 4096), (4096, 1))
    assert_size_stride(primals_99, (1024, ), (1, ))
    assert_size_stride(primals_100, (1024, ), (1, ))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, 1024), (1024, 1))
    assert_size_stride(primals_103, (1024, ), (1, ))
    assert_size_stride(primals_104, (1024, 1024), (1024, 1))
    assert_size_stride(primals_105, (1024, ), (1, ))
    assert_size_stride(primals_106, (1024, 1024), (1024, 1))
    assert_size_stride(primals_107, (1024, ), (1, ))
    assert_size_stride(primals_108, (1024, 1024), (1024, 1))
    assert_size_stride(primals_109, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (4096, 1024), (1024, 1))
    assert_size_stride(primals_113, (4096, ), (1, ))
    assert_size_stride(primals_114, (1024, 4096), (4096, 1))
    assert_size_stride(primals_115, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_118, (1024, 1024), (1024, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, 1024), (1024, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (1024, 1024), (1024, 1))
    assert_size_stride(primals_123, (1024, ), (1, ))
    assert_size_stride(primals_124, (1024, 1024), (1024, 1))
    assert_size_stride(primals_125, (1024, ), (1, ))
    assert_size_stride(primals_126, (1024, ), (1, ))
    assert_size_stride(primals_127, (1024, ), (1, ))
    assert_size_stride(primals_128, (4096, 1024), (1024, 1))
    assert_size_stride(primals_129, (4096, ), (1, ))
    assert_size_stride(primals_130, (1024, 4096), (4096, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (1024, 1024), (1024, 1))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, 1024), (1024, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_138, (1024, 1024), (1024, 1))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (1024, 1024), (1024, 1))
    assert_size_stride(primals_141, (1024, ), (1, ))
    assert_size_stride(primals_142, (1024, ), (1, ))
    assert_size_stride(primals_143, (1024, ), (1, ))
    assert_size_stride(primals_144, (4096, 1024), (1024, 1))
    assert_size_stride(primals_145, (4096, ), (1, ))
    assert_size_stride(primals_146, (1024, 4096), (4096, 1))
    assert_size_stride(primals_147, (1024, ), (1, ))
    assert_size_stride(primals_148, (1024, ), (1, ))
    assert_size_stride(primals_149, (1024, ), (1, ))
    assert_size_stride(primals_150, (1024, 1024), (1024, 1))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_152, (1024, 1024), (1024, 1))
    assert_size_stride(primals_153, (1024, ), (1, ))
    assert_size_stride(primals_154, (1024, 1024), (1024, 1))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_156, (1024, 1024), (1024, 1))
    assert_size_stride(primals_157, (1024, ), (1, ))
    assert_size_stride(primals_158, (1024, ), (1, ))
    assert_size_stride(primals_159, (1024, ), (1, ))
    assert_size_stride(primals_160, (4096, 1024), (1024, 1))
    assert_size_stride(primals_161, (4096, ), (1, ))
    assert_size_stride(primals_162, (1024, 4096), (4096, 1))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_165, (1024, ), (1, ))
    assert_size_stride(primals_166, (1024, 1024), (1024, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, 1024), (1024, 1))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (1024, 1024), (1024, 1))
    assert_size_stride(primals_171, (1024, ), (1, ))
    assert_size_stride(primals_172, (1024, 1024), (1024, 1))
    assert_size_stride(primals_173, (1024, ), (1, ))
    assert_size_stride(primals_174, (1024, ), (1, ))
    assert_size_stride(primals_175, (1024, ), (1, ))
    assert_size_stride(primals_176, (4096, 1024), (1024, 1))
    assert_size_stride(primals_177, (4096, ), (1, ))
    assert_size_stride(primals_178, (1024, 4096), (4096, 1))
    assert_size_stride(primals_179, (1024, ), (1, ))
    assert_size_stride(primals_180, (1024, ), (1, ))
    assert_size_stride(primals_181, (1024, ), (1, ))
    assert_size_stride(primals_182, (1024, 1024), (1024, 1))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (1024, 1024), (1024, 1))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, 1024), (1024, 1))
    assert_size_stride(primals_187, (1024, ), (1, ))
    assert_size_stride(primals_188, (1024, 1024), (1024, 1))
    assert_size_stride(primals_189, (1024, ), (1, ))
    assert_size_stride(primals_190, (1024, ), (1, ))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_192, (4096, 1024), (1024, 1))
    assert_size_stride(primals_193, (4096, ), (1, ))
    assert_size_stride(primals_194, (1024, 4096), (4096, 1))
    assert_size_stride(primals_195, (1024, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (1024, 1024), (1024, 1))
    assert_size_stride(primals_199, (1024, ), (1, ))
    assert_size_stride(primals_200, (1024, 1024), (1024, 1))
    assert_size_stride(primals_201, (1024, ), (1, ))
    assert_size_stride(primals_202, (1024, 1024), (1024, 1))
    assert_size_stride(primals_203, (1024, ), (1, ))
    assert_size_stride(primals_204, (1024, 1024), (1024, 1))
    assert_size_stride(primals_205, (1024, ), (1, ))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_208, (4096, 1024), (1024, 1))
    assert_size_stride(primals_209, (4096, ), (1, ))
    assert_size_stride(primals_210, (1024, 4096), (4096, 1))
    assert_size_stride(primals_211, (1024, ), (1, ))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_213, (1024, ), (1, ))
    assert_size_stride(primals_214, (1024, 1024), (1024, 1))
    assert_size_stride(primals_215, (1024, ), (1, ))
    assert_size_stride(primals_216, (1024, 1024), (1024, 1))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_218, (1024, 1024), (1024, 1))
    assert_size_stride(primals_219, (1024, ), (1, ))
    assert_size_stride(primals_220, (1024, 1024), (1024, 1))
    assert_size_stride(primals_221, (1024, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_224, (4096, 1024), (1024, 1))
    assert_size_stride(primals_225, (4096, ), (1, ))
    assert_size_stride(primals_226, (1024, 4096), (4096, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (1024, ), (1, ))
    assert_size_stride(primals_230, (1024, 1024), (1024, 1))
    assert_size_stride(primals_231, (1024, ), (1, ))
    assert_size_stride(primals_232, (1024, 1024), (1024, 1))
    assert_size_stride(primals_233, (1024, ), (1, ))
    assert_size_stride(primals_234, (1024, 1024), (1024, 1))
    assert_size_stride(primals_235, (1024, ), (1, ))
    assert_size_stride(primals_236, (1024, 1024), (1024, 1))
    assert_size_stride(primals_237, (1024, ), (1, ))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_239, (1024, ), (1, ))
    assert_size_stride(primals_240, (4096, 1024), (1024, 1))
    assert_size_stride(primals_241, (4096, ), (1, ))
    assert_size_stride(primals_242, (1024, 4096), (4096, 1))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (1024, ), (1, ))
    assert_size_stride(primals_246, (1024, 1024), (1024, 1))
    assert_size_stride(primals_247, (1024, ), (1, ))
    assert_size_stride(primals_248, (1024, 1024), (1024, 1))
    assert_size_stride(primals_249, (1024, ), (1, ))
    assert_size_stride(primals_250, (1024, 1024), (1024, 1))
    assert_size_stride(primals_251, (1024, ), (1, ))
    assert_size_stride(primals_252, (1024, 1024), (1024, 1))
    assert_size_stride(primals_253, (1024, ), (1, ))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_255, (1024, ), (1, ))
    assert_size_stride(primals_256, (4096, 1024), (1024, 1))
    assert_size_stride(primals_257, (4096, ), (1, ))
    assert_size_stride(primals_258, (1024, 4096), (4096, 1))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (1024, 1024), (1024, 1))
    assert_size_stride(primals_263, (1024, ), (1, ))
    assert_size_stride(primals_264, (1024, 1024), (1024, 1))
    assert_size_stride(primals_265, (1024, ), (1, ))
    assert_size_stride(primals_266, (1024, 1024), (1024, 1))
    assert_size_stride(primals_267, (1024, ), (1, ))
    assert_size_stride(primals_268, (1024, 1024), (1024, 1))
    assert_size_stride(primals_269, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (4096, 1024), (1024, 1))
    assert_size_stride(primals_273, (4096, ), (1, ))
    assert_size_stride(primals_274, (1024, 4096), (4096, 1))
    assert_size_stride(primals_275, (1024, ), (1, ))
    assert_size_stride(primals_276, (1024, ), (1, ))
    assert_size_stride(primals_277, (1024, ), (1, ))
    assert_size_stride(primals_278, (1024, 1024), (1024, 1))
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, 1024), (1024, 1))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, 1024), (1024, 1))
    assert_size_stride(primals_283, (1024, ), (1, ))
    assert_size_stride(primals_284, (1024, 1024), (1024, 1))
    assert_size_stride(primals_285, (1024, ), (1, ))
    assert_size_stride(primals_286, (1024, ), (1, ))
    assert_size_stride(primals_287, (1024, ), (1, ))
    assert_size_stride(primals_288, (4096, 1024), (1024, 1))
    assert_size_stride(primals_289, (4096, ), (1, ))
    assert_size_stride(primals_290, (1024, 4096), (4096, 1))
    assert_size_stride(primals_291, (1024, ), (1, ))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_293, (1024, ), (1, ))
    assert_size_stride(primals_294, (1024, 1024), (1024, 1))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_296, (1024, 1024), (1024, 1))
    assert_size_stride(primals_297, (1024, ), (1, ))
    assert_size_stride(primals_298, (1024, 1024), (1024, 1))
    assert_size_stride(primals_299, (1024, ), (1, ))
    assert_size_stride(primals_300, (1024, 1024), (1024, 1))
    assert_size_stride(primals_301, (1024, ), (1, ))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (4096, 1024), (1024, 1))
    assert_size_stride(primals_305, (4096, ), (1, ))
    assert_size_stride(primals_306, (1024, 4096), (4096, 1))
    assert_size_stride(primals_307, (1024, ), (1, ))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_309, (1024, ), (1, ))
    assert_size_stride(primals_310, (1024, 1024), (1024, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, 1024), (1024, 1))
    assert_size_stride(primals_313, (1024, ), (1, ))
    assert_size_stride(primals_314, (1024, 1024), (1024, 1))
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (1024, 1024), (1024, 1))
    assert_size_stride(primals_317, (1024, ), (1, ))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_319, (1024, ), (1, ))
    assert_size_stride(primals_320, (4096, 1024), (1024, 1))
    assert_size_stride(primals_321, (4096, ), (1, ))
    assert_size_stride(primals_322, (1024, 4096), (4096, 1))
    assert_size_stride(primals_323, (1024, ), (1, ))
    assert_size_stride(primals_324, (1024, ), (1, ))
    assert_size_stride(primals_325, (1024, ), (1, ))
    assert_size_stride(primals_326, (1024, 1024), (1024, 1))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_328, (1024, 1024), (1024, 1))
    assert_size_stride(primals_329, (1024, ), (1, ))
    assert_size_stride(primals_330, (1024, 1024), (1024, 1))
    assert_size_stride(primals_331, (1024, ), (1, ))
    assert_size_stride(primals_332, (1024, 1024), (1024, 1))
    assert_size_stride(primals_333, (1024, ), (1, ))
    assert_size_stride(primals_334, (1024, ), (1, ))
    assert_size_stride(primals_335, (1024, ), (1, ))
    assert_size_stride(primals_336, (4096, 1024), (1024, 1))
    assert_size_stride(primals_337, (4096, ), (1, ))
    assert_size_stride(primals_338, (1024, 4096), (4096, 1))
    assert_size_stride(primals_339, (1024, ), (1, ))
    assert_size_stride(primals_340, (1024, ), (1, ))
    assert_size_stride(primals_341, (1024, ), (1, ))
    assert_size_stride(primals_342, (1024, 1024), (1024, 1))
    assert_size_stride(primals_343, (1024, ), (1, ))
    assert_size_stride(primals_344, (1024, 1024), (1024, 1))
    assert_size_stride(primals_345, (1024, ), (1, ))
    assert_size_stride(primals_346, (1024, 1024), (1024, 1))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_348, (1024, 1024), (1024, 1))
    assert_size_stride(primals_349, (1024, ), (1, ))
    assert_size_stride(primals_350, (1024, ), (1, ))
    assert_size_stride(primals_351, (1024, ), (1, ))
    assert_size_stride(primals_352, (4096, 1024), (1024, 1))
    assert_size_stride(primals_353, (4096, ), (1, ))
    assert_size_stride(primals_354, (1024, 4096), (4096, 1))
    assert_size_stride(primals_355, (1024, ), (1, ))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_357, (1024, ), (1, ))
    assert_size_stride(primals_358, (1024, 1024), (1024, 1))
    assert_size_stride(primals_359, (1024, ), (1, ))
    assert_size_stride(primals_360, (1024, 1024), (1024, 1))
    assert_size_stride(primals_361, (1024, ), (1, ))
    assert_size_stride(primals_362, (1024, 1024), (1024, 1))
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_364, (1024, 1024), (1024, 1))
    assert_size_stride(primals_365, (1024, ), (1, ))
    assert_size_stride(primals_366, (1024, ), (1, ))
    assert_size_stride(primals_367, (1024, ), (1, ))
    assert_size_stride(primals_368, (4096, 1024), (1024, 1))
    assert_size_stride(primals_369, (4096, ), (1, ))
    assert_size_stride(primals_370, (1024, 4096), (4096, 1))
    assert_size_stride(primals_371, (1024, ), (1, ))
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_373, (1024, ), (1, ))
    assert_size_stride(primals_374, (1024, 1024), (1024, 1))
    assert_size_stride(primals_375, (1024, ), (1, ))
    assert_size_stride(primals_376, (1024, 1024), (1024, 1))
    assert_size_stride(primals_377, (1024, ), (1, ))
    assert_size_stride(primals_378, (1024, 1024), (1024, 1))
    assert_size_stride(primals_379, (1024, ), (1, ))
    assert_size_stride(primals_380, (1024, 1024), (1024, 1))
    assert_size_stride(primals_381, (1024, ), (1, ))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_383, (1024, ), (1, ))
    assert_size_stride(primals_384, (4096, 1024), (1024, 1))
    assert_size_stride(primals_385, (4096, ), (1, ))
    assert_size_stride(primals_386, (1024, 4096), (4096, 1))
    assert_size_stride(primals_387, (1024, ), (1, ))
    assert_size_stride(primals_388, (1024, ), (1, ))
    assert_size_stride(primals_389, (1024, ), (1, ))
    assert_size_stride(primals_390, (2, 1024), (1024, 1))
    assert_size_stride(primals_391, (2, ), (1, ))
    assert_size_stride(primals_392, (1, 512), (512, 1))
    assert_size_stride(primals_393, (1, 512), (512, 1))
    assert_size_stride(primals_394, (1, ), (1, ))
    assert_size_stride(primals_395, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_zeros_0.run(buf0, 512, grid=grid(512), stream=stream0)
        buf1 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding]
        triton_poi_fused_add_embedding_1.run(primals_393, primals_1, primals_2, primals_392, primals_3, buf1, 524288, grid=grid(524288), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        # Source Nodes: [embedding_output, embeddings, embeddings_1, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_dropout]
        buf2 = aten.native_dropout(buf1, 0.1, True)
        buf3 = buf2[0]
        buf4 = buf2[1]
        del buf2
        buf8 = buf1; del buf1  # reuse
        buf9 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf853 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [ln_outputs, mixed_query_layer], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_2.run(buf3, primals_4, primals_5, buf8, buf9, buf853, 512, 1024, grid=grid(512), stream=stream0)
        del primals_5
        buf10 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(primals_6, (1024, 1024), (1, 1024), 0), out=buf10)
        buf11 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(primals_8, (1024, 1024), (1, 1024), 0), out=buf11)
        buf12 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(primals_10, (1024, 1024), (1, 1024), 0), out=buf12)
        buf13 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf10, primals_7, buf13, 524288, grid=grid(524288), stream=stream0)
        del primals_7
        buf14 = reinterpret_tensor(buf10, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf11, primals_9, buf14, 524288, grid=grid(524288), stream=stream0)
        del primals_9
        buf15 = reinterpret_tensor(buf11, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf12, primals_11, buf15, 524288, grid=grid(524288), stream=stream0)
        del primals_11
        # Source Nodes: [], Original ATen: []
        buf16 = aten._scaled_dot_product_efficient_attention(buf13, buf14, buf15, None, True, 0.1, scale=0.125)
        buf17 = buf16[0]
        buf18 = buf16[1]
        buf19 = buf16[2]
        buf20 = buf16[3]
        del buf16
        buf21 = buf12; del buf12  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf17, buf21, 524288, grid=grid(524288), stream=stream0)
        buf22 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf21, reinterpret_tensor(primals_12, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf22)
        del primals_13
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_dropout]
        buf23 = aten.native_dropout(reinterpret_tensor(buf22, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf24 = buf23[0]
        buf25 = buf23[1]
        del buf23
        buf29 = reinterpret_tensor(buf22, (1, 512, 1024), (524288, 1024, 1), 0); del buf22  # reuse
        buf30 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf852 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output, hidden_states_2, ln_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf3, buf24, primals_14, primals_15, buf29, buf30, buf852, 512, 1024, grid=grid(512), stream=stream0)
        del primals_15
        buf31 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf30, reinterpret_tensor(primals_16, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf31)
        del primals_17
        buf32 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_4, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf31, buf32, 2097152, grid=grid(2097152), stream=stream0)
        buf33 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf32, reinterpret_tensor(primals_18, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf33)
        del primals_19
        # Source Nodes: [hidden_states_5], Original ATen: [aten.native_dropout]
        buf34 = aten.native_dropout(reinterpret_tensor(buf33, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf40 = reinterpret_tensor(buf33, (1, 512, 1024), (524288, 1024, 1), 0); del buf33  # reuse
        buf41 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf851 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output, hidden_states_6, ln_outputs_1, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf3, buf24, buf35, primals_20, primals_21, buf40, buf41, buf851, 512, 1024, grid=grid(512), stream=stream0)
        del primals_21
        buf42 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_22, (1024, 1024), (1, 1024), 0), out=buf42)
        buf43 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_24, (1024, 1024), (1, 1024), 0), out=buf43)
        buf44 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_26, (1024, 1024), (1, 1024), 0), out=buf44)
        buf45 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf42, primals_23, buf45, 524288, grid=grid(524288), stream=stream0)
        del primals_23
        buf46 = reinterpret_tensor(buf42, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf43, primals_25, buf46, 524288, grid=grid(524288), stream=stream0)
        del primals_25
        buf47 = reinterpret_tensor(buf43, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf44, primals_27, buf47, 524288, grid=grid(524288), stream=stream0)
        del primals_27
        # Source Nodes: [], Original ATen: []
        buf48 = aten._scaled_dot_product_efficient_attention(buf45, buf46, buf47, None, True, 0.1, scale=0.125)
        buf49 = buf48[0]
        buf50 = buf48[1]
        buf51 = buf48[2]
        buf52 = buf48[3]
        del buf48
        buf53 = buf44; del buf44  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf49, buf53, 524288, grid=grid(524288), stream=stream0)
        buf54 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, buf53, reinterpret_tensor(primals_28, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf54)
        del primals_29
        # Source Nodes: [hidden_states_8], Original ATen: [aten.native_dropout]
        buf55 = aten.native_dropout(reinterpret_tensor(buf54, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf61 = reinterpret_tensor(buf54, (1, 512, 1024), (524288, 1024, 1), 0); del buf54  # reuse
        buf62 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf850 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output, attention_output_2, hidden_states_6, hidden_states_9, ln_output_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf3, buf24, buf35, buf56, primals_30, primals_31, buf61, buf62, buf850, 512, 1024, grid=grid(512), stream=stream0)
        del primals_31
        buf63 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf62, reinterpret_tensor(primals_32, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf63)
        del primals_33
        buf64 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_11, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf63, buf64, 2097152, grid=grid(2097152), stream=stream0)
        buf65 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, buf64, reinterpret_tensor(primals_34, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf65)
        del primals_35
        # Source Nodes: [hidden_states_12], Original ATen: [aten.native_dropout]
        buf66 = aten.native_dropout(reinterpret_tensor(buf65, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf67 = buf66[0]
        buf68 = buf66[1]
        del buf66
        buf69 = buf67; del buf67  # reuse
        buf73 = reinterpret_tensor(buf65, (1, 512, 1024), (524288, 1024, 1), 0); del buf65  # reuse
        buf74 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf849 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output, attention_output_2, hidden_states_13, hidden_states_6, ln_outputs_2, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf69, buf3, buf24, buf35, buf56, primals_36, primals_37, buf73, buf74, buf849, 512, 1024, grid=grid(512), stream=stream0)
        del primals_37
        buf75 = reinterpret_tensor(buf56, (512, 1024), (1024, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf74, reinterpret_tensor(primals_38, (1024, 1024), (1, 1024), 0), out=buf75)
        buf76 = reinterpret_tensor(buf35, (512, 1024), (1024, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf74, reinterpret_tensor(primals_40, (1024, 1024), (1, 1024), 0), out=buf76)
        buf77 = reinterpret_tensor(buf3, (512, 1024), (1024, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf74, reinterpret_tensor(primals_42, (1024, 1024), (1, 1024), 0), out=buf77)
        buf78 = reinterpret_tensor(buf24, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf75, primals_39, buf78, 524288, grid=grid(524288), stream=stream0)
        del primals_39
        buf79 = reinterpret_tensor(buf75, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf76, primals_41, buf79, 524288, grid=grid(524288), stream=stream0)
        del primals_41
        buf80 = reinterpret_tensor(buf76, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf77, primals_43, buf80, 524288, grid=grid(524288), stream=stream0)
        del primals_43
        # Source Nodes: [], Original ATen: []
        buf81 = aten._scaled_dot_product_efficient_attention(buf78, buf79, buf80, None, True, 0.1, scale=0.125)
        buf82 = buf81[0]
        buf83 = buf81[1]
        buf84 = buf81[2]
        buf85 = buf81[3]
        del buf81
        buf86 = buf77; del buf77  # reuse
        # Source Nodes: [hidden_states_14], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf82, buf86, 524288, grid=grid(524288), stream=stream0)
        buf87 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, buf86, reinterpret_tensor(primals_44, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf87)
        del primals_45
        # Source Nodes: [hidden_states_15], Original ATen: [aten.native_dropout]
        buf88 = aten.native_dropout(reinterpret_tensor(buf87, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf89 = buf88[0]
        buf90 = buf88[1]
        del buf88
        buf94 = reinterpret_tensor(buf87, (1, 512, 1024), (524288, 1024, 1), 0); del buf87  # reuse
        buf95 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf848 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_4, hidden_states_16, ln_output_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf69, buf89, primals_46, primals_47, buf94, buf95, buf848, 512, 1024, grid=grid(512), stream=stream0)
        del primals_47
        buf96 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf95, reinterpret_tensor(primals_48, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf96)
        del primals_49
        buf97 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf96, buf97, 2097152, grid=grid(2097152), stream=stream0)
        buf98 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, buf97, reinterpret_tensor(primals_50, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf98)
        del primals_51
        # Source Nodes: [hidden_states_19], Original ATen: [aten.native_dropout]
        buf99 = aten.native_dropout(reinterpret_tensor(buf98, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf105 = reinterpret_tensor(buf98, (1, 512, 1024), (524288, 1024, 1), 0); del buf98  # reuse
        buf106 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf847 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_4, hidden_states_20, ln_outputs_3, mixed_query_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf69, buf89, buf100, primals_52, primals_53, buf105, buf106, buf847, 512, 1024, grid=grid(512), stream=stream0)
        del primals_53
        buf107 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf106, reinterpret_tensor(primals_54, (1024, 1024), (1, 1024), 0), out=buf107)
        buf108 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf106, reinterpret_tensor(primals_56, (1024, 1024), (1, 1024), 0), out=buf108)
        buf109 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf106, reinterpret_tensor(primals_58, (1024, 1024), (1, 1024), 0), out=buf109)
        buf110 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf107, primals_55, buf110, 524288, grid=grid(524288), stream=stream0)
        del primals_55
        buf111 = reinterpret_tensor(buf107, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf108, primals_57, buf111, 524288, grid=grid(524288), stream=stream0)
        del primals_57
        buf112 = reinterpret_tensor(buf108, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf109, primals_59, buf112, 524288, grid=grid(524288), stream=stream0)
        del primals_59
        # Source Nodes: [], Original ATen: []
        buf113 = aten._scaled_dot_product_efficient_attention(buf110, buf111, buf112, None, True, 0.1, scale=0.125)
        buf114 = buf113[0]
        buf115 = buf113[1]
        buf116 = buf113[2]
        buf117 = buf113[3]
        del buf113
        buf118 = buf109; del buf109  # reuse
        # Source Nodes: [hidden_states_21], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf114, buf118, 524288, grid=grid(524288), stream=stream0)
        buf119 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, buf118, reinterpret_tensor(primals_60, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf119)
        del primals_61
        # Source Nodes: [hidden_states_22], Original ATen: [aten.native_dropout]
        buf120 = aten.native_dropout(reinterpret_tensor(buf119, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf121 = buf120[0]
        buf122 = buf120[1]
        del buf120
        buf126 = reinterpret_tensor(buf119, (1, 512, 1024), (524288, 1024, 1), 0); del buf119  # reuse
        buf127 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf846 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_4, attention_output_6, hidden_states_20, hidden_states_23, ln_output_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf69, buf89, buf100, buf121, primals_62, primals_63, buf126, buf127, buf846, 512, 1024, grid=grid(512), stream=stream0)
        del primals_63
        buf128 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf127, reinterpret_tensor(primals_64, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf128)
        del primals_65
        buf129 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_25, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf128, buf129, 2097152, grid=grid(2097152), stream=stream0)
        buf130 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_67, buf129, reinterpret_tensor(primals_66, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf130)
        del primals_67
        # Source Nodes: [hidden_states_26], Original ATen: [aten.native_dropout]
        buf131 = aten.native_dropout(reinterpret_tensor(buf130, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf132 = buf131[0]
        buf133 = buf131[1]
        del buf131
        buf134 = buf132; del buf132  # reuse
        buf138 = reinterpret_tensor(buf130, (1, 512, 1024), (524288, 1024, 1), 0); del buf130  # reuse
        buf139 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf845 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_4, attention_output_6, hidden_states_20, hidden_states_27, ln_outputs_4, mixed_query_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf134, buf69, buf89, buf100, buf121, primals_68, primals_69, buf138, buf139, buf845, 512, 1024, grid=grid(512), stream=stream0)
        del primals_69
        buf140 = reinterpret_tensor(buf89, (512, 1024), (1024, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf139, reinterpret_tensor(primals_70, (1024, 1024), (1, 1024), 0), out=buf140)
        buf141 = reinterpret_tensor(buf69, (512, 1024), (1024, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf139, reinterpret_tensor(primals_72, (1024, 1024), (1, 1024), 0), out=buf141)
        buf142 = reinterpret_tensor(buf121, (512, 1024), (1024, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf139, reinterpret_tensor(primals_74, (1024, 1024), (1, 1024), 0), out=buf142)
        buf143 = reinterpret_tensor(buf100, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf140, primals_71, buf143, 524288, grid=grid(524288), stream=stream0)
        del primals_71
        buf144 = reinterpret_tensor(buf140, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf141, primals_73, buf144, 524288, grid=grid(524288), stream=stream0)
        del primals_73
        buf145 = reinterpret_tensor(buf141, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf142, primals_75, buf145, 524288, grid=grid(524288), stream=stream0)
        del primals_75
        # Source Nodes: [], Original ATen: []
        buf146 = aten._scaled_dot_product_efficient_attention(buf143, buf144, buf145, None, True, 0.1, scale=0.125)
        buf147 = buf146[0]
        buf148 = buf146[1]
        buf149 = buf146[2]
        buf150 = buf146[3]
        del buf146
        buf151 = buf142; del buf142  # reuse
        # Source Nodes: [hidden_states_28], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf147, buf151, 524288, grid=grid(524288), stream=stream0)
        buf152 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, buf151, reinterpret_tensor(primals_76, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf152)
        del primals_77
        # Source Nodes: [hidden_states_29], Original ATen: [aten.native_dropout]
        buf153 = aten.native_dropout(reinterpret_tensor(buf152, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf154 = buf153[0]
        buf155 = buf153[1]
        del buf153
        buf159 = reinterpret_tensor(buf152, (1, 512, 1024), (524288, 1024, 1), 0); del buf152  # reuse
        buf160 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf844 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_8, hidden_states_30, ln_output_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf134, buf154, primals_78, primals_79, buf159, buf160, buf844, 512, 1024, grid=grid(512), stream=stream0)
        del primals_79
        buf161 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf160, reinterpret_tensor(primals_80, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf161)
        del primals_81
        buf162 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf161, buf162, 2097152, grid=grid(2097152), stream=stream0)
        buf163 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_83, buf162, reinterpret_tensor(primals_82, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf163)
        del primals_83
        # Source Nodes: [hidden_states_33], Original ATen: [aten.native_dropout]
        buf164 = aten.native_dropout(reinterpret_tensor(buf163, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf165 = buf164[0]
        buf166 = buf164[1]
        del buf164
        buf170 = reinterpret_tensor(buf163, (1, 512, 1024), (524288, 1024, 1), 0); del buf163  # reuse
        buf171 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf843 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_8, hidden_states_34, ln_outputs_5, mixed_query_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf134, buf154, buf165, primals_84, primals_85, buf170, buf171, buf843, 512, 1024, grid=grid(512), stream=stream0)
        del primals_85
        buf172 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, reinterpret_tensor(primals_86, (1024, 1024), (1, 1024), 0), out=buf172)
        buf173 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, reinterpret_tensor(primals_88, (1024, 1024), (1, 1024), 0), out=buf173)
        buf174 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, reinterpret_tensor(primals_90, (1024, 1024), (1, 1024), 0), out=buf174)
        buf175 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf172, primals_87, buf175, 524288, grid=grid(524288), stream=stream0)
        del primals_87
        buf176 = reinterpret_tensor(buf172, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf173, primals_89, buf176, 524288, grid=grid(524288), stream=stream0)
        del primals_89
        buf177 = reinterpret_tensor(buf173, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf174, primals_91, buf177, 524288, grid=grid(524288), stream=stream0)
        del primals_91
        # Source Nodes: [], Original ATen: []
        buf178 = aten._scaled_dot_product_efficient_attention(buf175, buf176, buf177, None, True, 0.1, scale=0.125)
        buf179 = buf178[0]
        buf180 = buf178[1]
        buf181 = buf178[2]
        buf182 = buf178[3]
        del buf178
        buf183 = buf174; del buf174  # reuse
        # Source Nodes: [hidden_states_35], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf179, buf183, 524288, grid=grid(524288), stream=stream0)
        buf184 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_35], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, buf183, reinterpret_tensor(primals_92, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf184)
        del primals_93
        # Source Nodes: [hidden_states_36], Original ATen: [aten.native_dropout]
        buf185 = aten.native_dropout(reinterpret_tensor(buf184, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf186 = buf185[0]
        buf187 = buf185[1]
        del buf185
        buf191 = reinterpret_tensor(buf184, (1, 512, 1024), (524288, 1024, 1), 0); del buf184  # reuse
        buf192 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf842 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_10, attention_output_8, hidden_states_34, hidden_states_37, ln_output_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf134, buf154, buf165, buf186, primals_94, primals_95, buf191, buf192, buf842, 512, 1024, grid=grid(512), stream=stream0)
        del primals_95
        buf193 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf192, reinterpret_tensor(primals_96, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf193)
        del primals_97
        buf194 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf193, buf194, 2097152, grid=grid(2097152), stream=stream0)
        buf195 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, buf194, reinterpret_tensor(primals_98, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf195)
        del primals_99
        # Source Nodes: [hidden_states_40], Original ATen: [aten.native_dropout]
        buf196 = aten.native_dropout(reinterpret_tensor(buf195, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf197 = buf196[0]
        buf198 = buf196[1]
        del buf196
        buf199 = buf197; del buf197  # reuse
        buf203 = reinterpret_tensor(buf195, (1, 512, 1024), (524288, 1024, 1), 0); del buf195  # reuse
        buf204 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf841 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_10, attention_output_8, hidden_states_34, hidden_states_41, ln_outputs_6, mixed_query_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf199, buf134, buf154, buf165, buf186, primals_100, primals_101, buf203, buf204, buf841, 512, 1024, grid=grid(512), stream=stream0)
        del primals_101
        buf205 = reinterpret_tensor(buf186, (512, 1024), (1024, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf204, reinterpret_tensor(primals_102, (1024, 1024), (1, 1024), 0), out=buf205)
        buf206 = reinterpret_tensor(buf165, (512, 1024), (1024, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf204, reinterpret_tensor(primals_104, (1024, 1024), (1, 1024), 0), out=buf206)
        buf207 = reinterpret_tensor(buf154, (512, 1024), (1024, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf204, reinterpret_tensor(primals_106, (1024, 1024), (1, 1024), 0), out=buf207)
        buf208 = reinterpret_tensor(buf134, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf205, primals_103, buf208, 524288, grid=grid(524288), stream=stream0)
        del primals_103
        buf209 = reinterpret_tensor(buf205, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf206, primals_105, buf209, 524288, grid=grid(524288), stream=stream0)
        del primals_105
        buf210 = reinterpret_tensor(buf206, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf207, primals_107, buf210, 524288, grid=grid(524288), stream=stream0)
        del primals_107
        # Source Nodes: [], Original ATen: []
        buf211 = aten._scaled_dot_product_efficient_attention(buf208, buf209, buf210, None, True, 0.1, scale=0.125)
        buf212 = buf211[0]
        buf213 = buf211[1]
        buf214 = buf211[2]
        buf215 = buf211[3]
        del buf211
        buf216 = buf207; del buf207  # reuse
        # Source Nodes: [hidden_states_42], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf212, buf216, 524288, grid=grid(524288), stream=stream0)
        buf217 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_109, buf216, reinterpret_tensor(primals_108, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf217)
        del primals_109
        # Source Nodes: [hidden_states_43], Original ATen: [aten.native_dropout]
        buf218 = aten.native_dropout(reinterpret_tensor(buf217, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf224 = reinterpret_tensor(buf217, (1, 512, 1024), (524288, 1024, 1), 0); del buf217  # reuse
        buf225 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf840 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_12, hidden_states_44, ln_output_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf199, buf219, primals_110, primals_111, buf224, buf225, buf840, 512, 1024, grid=grid(512), stream=stream0)
        del primals_111
        buf226 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_113, buf225, reinterpret_tensor(primals_112, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf226)
        del primals_113
        buf227 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_46, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf226, buf227, 2097152, grid=grid(2097152), stream=stream0)
        buf228 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_46], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_115, buf227, reinterpret_tensor(primals_114, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf228)
        del primals_115
        # Source Nodes: [hidden_states_47], Original ATen: [aten.native_dropout]
        buf229 = aten.native_dropout(reinterpret_tensor(buf228, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf235 = reinterpret_tensor(buf228, (1, 512, 1024), (524288, 1024, 1), 0); del buf228  # reuse
        buf236 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf839 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_12, hidden_states_48, ln_outputs_7, mixed_query_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf199, buf219, buf230, primals_116, primals_117, buf235, buf236, buf839, 512, 1024, grid=grid(512), stream=stream0)
        del primals_117
        buf237 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_118, (1024, 1024), (1, 1024), 0), out=buf237)
        buf238 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_120, (1024, 1024), (1, 1024), 0), out=buf238)
        buf239 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_122, (1024, 1024), (1, 1024), 0), out=buf239)
        buf240 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf237, primals_119, buf240, 524288, grid=grid(524288), stream=stream0)
        del primals_119
        buf241 = reinterpret_tensor(buf237, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf238, primals_121, buf241, 524288, grid=grid(524288), stream=stream0)
        del primals_121
        buf242 = reinterpret_tensor(buf238, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf239, primals_123, buf242, 524288, grid=grid(524288), stream=stream0)
        del primals_123
        # Source Nodes: [], Original ATen: []
        buf243 = aten._scaled_dot_product_efficient_attention(buf240, buf241, buf242, None, True, 0.1, scale=0.125)
        buf244 = buf243[0]
        buf245 = buf243[1]
        buf246 = buf243[2]
        buf247 = buf243[3]
        del buf243
        buf248 = buf239; del buf239  # reuse
        # Source Nodes: [hidden_states_49], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf244, buf248, 524288, grid=grid(524288), stream=stream0)
        buf249 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_125, buf248, reinterpret_tensor(primals_124, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf249)
        del primals_125
        # Source Nodes: [hidden_states_50], Original ATen: [aten.native_dropout]
        buf250 = aten.native_dropout(reinterpret_tensor(buf249, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf256 = reinterpret_tensor(buf249, (1, 512, 1024), (524288, 1024, 1), 0); del buf249  # reuse
        buf257 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf838 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_12, attention_output_14, hidden_states_48, hidden_states_51, ln_output_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf199, buf219, buf230, buf251, primals_126, primals_127, buf256, buf257, buf838, 512, 1024, grid=grid(512), stream=stream0)
        del primals_127
        buf258 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf257, reinterpret_tensor(primals_128, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf258)
        del primals_129
        buf259 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_53, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf258, buf259, 2097152, grid=grid(2097152), stream=stream0)
        buf260 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_131, buf259, reinterpret_tensor(primals_130, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf260)
        del primals_131
        # Source Nodes: [hidden_states_54], Original ATen: [aten.native_dropout]
        buf261 = aten.native_dropout(reinterpret_tensor(buf260, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf262 = buf261[0]
        buf263 = buf261[1]
        del buf261
        buf264 = buf262; del buf262  # reuse
        buf268 = reinterpret_tensor(buf260, (1, 512, 1024), (524288, 1024, 1), 0); del buf260  # reuse
        buf269 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf837 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_12, attention_output_14, hidden_states_48, hidden_states_55, ln_outputs_8, mixed_query_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf264, buf199, buf219, buf230, buf251, primals_132, primals_133, buf268, buf269, buf837, 512, 1024, grid=grid(512), stream=stream0)
        del primals_133
        buf270 = reinterpret_tensor(buf251, (512, 1024), (1024, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf269, reinterpret_tensor(primals_134, (1024, 1024), (1, 1024), 0), out=buf270)
        buf271 = reinterpret_tensor(buf230, (512, 1024), (1024, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf269, reinterpret_tensor(primals_136, (1024, 1024), (1, 1024), 0), out=buf271)
        buf272 = reinterpret_tensor(buf219, (512, 1024), (1024, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf269, reinterpret_tensor(primals_138, (1024, 1024), (1, 1024), 0), out=buf272)
        buf273 = reinterpret_tensor(buf199, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf270, primals_135, buf273, 524288, grid=grid(524288), stream=stream0)
        del primals_135
        buf274 = reinterpret_tensor(buf270, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf271, primals_137, buf274, 524288, grid=grid(524288), stream=stream0)
        del primals_137
        buf275 = reinterpret_tensor(buf271, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf272, primals_139, buf275, 524288, grid=grid(524288), stream=stream0)
        del primals_139
        # Source Nodes: [], Original ATen: []
        buf276 = aten._scaled_dot_product_efficient_attention(buf273, buf274, buf275, None, True, 0.1, scale=0.125)
        buf277 = buf276[0]
        buf278 = buf276[1]
        buf279 = buf276[2]
        buf280 = buf276[3]
        del buf276
        buf281 = buf272; del buf272  # reuse
        # Source Nodes: [hidden_states_56], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf277, buf281, 524288, grid=grid(524288), stream=stream0)
        buf282 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_141, buf281, reinterpret_tensor(primals_140, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf282)
        del primals_141
        # Source Nodes: [hidden_states_57], Original ATen: [aten.native_dropout]
        buf283 = aten.native_dropout(reinterpret_tensor(buf282, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf284 = buf283[0]
        buf285 = buf283[1]
        del buf283
        buf289 = reinterpret_tensor(buf282, (1, 512, 1024), (524288, 1024, 1), 0); del buf282  # reuse
        buf290 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf836 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_16, hidden_states_58, ln_output_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf264, buf284, primals_142, primals_143, buf289, buf290, buf836, 512, 1024, grid=grid(512), stream=stream0)
        del primals_143
        buf291 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, buf290, reinterpret_tensor(primals_144, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf291)
        del primals_145
        buf292 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_60, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf291, buf292, 2097152, grid=grid(2097152), stream=stream0)
        buf293 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_147, buf292, reinterpret_tensor(primals_146, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf293)
        del primals_147
        # Source Nodes: [hidden_states_61], Original ATen: [aten.native_dropout]
        buf294 = aten.native_dropout(reinterpret_tensor(buf293, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf295 = buf294[0]
        buf296 = buf294[1]
        del buf294
        buf300 = reinterpret_tensor(buf293, (1, 512, 1024), (524288, 1024, 1), 0); del buf293  # reuse
        buf301 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf835 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_16, hidden_states_62, ln_outputs_9, mixed_query_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf264, buf284, buf295, primals_148, primals_149, buf300, buf301, buf835, 512, 1024, grid=grid(512), stream=stream0)
        del primals_149
        buf302 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf301, reinterpret_tensor(primals_150, (1024, 1024), (1, 1024), 0), out=buf302)
        buf303 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf301, reinterpret_tensor(primals_152, (1024, 1024), (1, 1024), 0), out=buf303)
        buf304 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf301, reinterpret_tensor(primals_154, (1024, 1024), (1, 1024), 0), out=buf304)
        buf305 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf302, primals_151, buf305, 524288, grid=grid(524288), stream=stream0)
        del primals_151
        buf306 = reinterpret_tensor(buf302, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf303, primals_153, buf306, 524288, grid=grid(524288), stream=stream0)
        del primals_153
        buf307 = reinterpret_tensor(buf303, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf304, primals_155, buf307, 524288, grid=grid(524288), stream=stream0)
        del primals_155
        # Source Nodes: [], Original ATen: []
        buf308 = aten._scaled_dot_product_efficient_attention(buf305, buf306, buf307, None, True, 0.1, scale=0.125)
        buf309 = buf308[0]
        buf310 = buf308[1]
        buf311 = buf308[2]
        buf312 = buf308[3]
        del buf308
        buf313 = buf304; del buf304  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf309, buf313, 524288, grid=grid(524288), stream=stream0)
        buf314 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_157, buf313, reinterpret_tensor(primals_156, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf314)
        del primals_157
        # Source Nodes: [hidden_states_64], Original ATen: [aten.native_dropout]
        buf315 = aten.native_dropout(reinterpret_tensor(buf314, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf316 = buf315[0]
        buf317 = buf315[1]
        del buf315
        buf321 = reinterpret_tensor(buf314, (1, 512, 1024), (524288, 1024, 1), 0); del buf314  # reuse
        buf322 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf834 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_16, attention_output_18, hidden_states_62, hidden_states_65, ln_output_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf264, buf284, buf295, buf316, primals_158, primals_159, buf321, buf322, buf834, 512, 1024, grid=grid(512), stream=stream0)
        del primals_159
        buf323 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_65], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf322, reinterpret_tensor(primals_160, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf323)
        del primals_161
        buf324 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_67, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf323, buf324, 2097152, grid=grid(2097152), stream=stream0)
        buf325 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_163, buf324, reinterpret_tensor(primals_162, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf325)
        del primals_163
        # Source Nodes: [hidden_states_68], Original ATen: [aten.native_dropout]
        buf326 = aten.native_dropout(reinterpret_tensor(buf325, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf327 = buf326[0]
        buf328 = buf326[1]
        del buf326
        buf329 = buf327; del buf327  # reuse
        buf333 = reinterpret_tensor(buf325, (1, 512, 1024), (524288, 1024, 1), 0); del buf325  # reuse
        buf334 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf833 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_16, attention_output_18, hidden_states_62, hidden_states_69, ln_outputs_10, mixed_query_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf329, buf264, buf284, buf295, buf316, primals_164, primals_165, buf333, buf334, buf833, 512, 1024, grid=grid(512), stream=stream0)
        del primals_165
        buf335 = reinterpret_tensor(buf316, (512, 1024), (1024, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf334, reinterpret_tensor(primals_166, (1024, 1024), (1, 1024), 0), out=buf335)
        buf336 = reinterpret_tensor(buf295, (512, 1024), (1024, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf334, reinterpret_tensor(primals_168, (1024, 1024), (1, 1024), 0), out=buf336)
        buf337 = reinterpret_tensor(buf284, (512, 1024), (1024, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf334, reinterpret_tensor(primals_170, (1024, 1024), (1, 1024), 0), out=buf337)
        buf338 = reinterpret_tensor(buf264, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf335, primals_167, buf338, 524288, grid=grid(524288), stream=stream0)
        del primals_167
        buf339 = reinterpret_tensor(buf335, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf336, primals_169, buf339, 524288, grid=grid(524288), stream=stream0)
        del primals_169
        buf340 = reinterpret_tensor(buf336, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf337, primals_171, buf340, 524288, grid=grid(524288), stream=stream0)
        del primals_171
        # Source Nodes: [], Original ATen: []
        buf341 = aten._scaled_dot_product_efficient_attention(buf338, buf339, buf340, None, True, 0.1, scale=0.125)
        buf342 = buf341[0]
        buf343 = buf341[1]
        buf344 = buf341[2]
        buf345 = buf341[3]
        del buf341
        buf346 = buf337; del buf337  # reuse
        # Source Nodes: [hidden_states_70], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf342, buf346, 524288, grid=grid(524288), stream=stream0)
        buf347 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf346, reinterpret_tensor(primals_172, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf347)
        del primals_173
        # Source Nodes: [hidden_states_71], Original ATen: [aten.native_dropout]
        buf348 = aten.native_dropout(reinterpret_tensor(buf347, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf349 = buf348[0]
        buf350 = buf348[1]
        del buf348
        buf354 = reinterpret_tensor(buf347, (1, 512, 1024), (524288, 1024, 1), 0); del buf347  # reuse
        buf355 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf832 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_20, hidden_states_72, ln_output_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf329, buf349, primals_174, primals_175, buf354, buf355, buf832, 512, 1024, grid=grid(512), stream=stream0)
        del primals_175
        buf356 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf355, reinterpret_tensor(primals_176, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf356)
        del primals_177
        buf357 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_74, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf356, buf357, 2097152, grid=grid(2097152), stream=stream0)
        buf358 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_179, buf357, reinterpret_tensor(primals_178, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf358)
        del primals_179
        # Source Nodes: [hidden_states_75], Original ATen: [aten.native_dropout]
        buf359 = aten.native_dropout(reinterpret_tensor(buf358, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf360 = buf359[0]
        buf361 = buf359[1]
        del buf359
        buf365 = reinterpret_tensor(buf358, (1, 512, 1024), (524288, 1024, 1), 0); del buf358  # reuse
        buf366 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf831 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_20, hidden_states_76, ln_outputs_11, mixed_query_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf329, buf349, buf360, primals_180, primals_181, buf365, buf366, buf831, 512, 1024, grid=grid(512), stream=stream0)
        del primals_181
        buf367 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_182, (1024, 1024), (1, 1024), 0), out=buf367)
        buf368 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_184, (1024, 1024), (1, 1024), 0), out=buf368)
        buf369 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_186, (1024, 1024), (1, 1024), 0), out=buf369)
        buf370 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf367, primals_183, buf370, 524288, grid=grid(524288), stream=stream0)
        del primals_183
        buf371 = reinterpret_tensor(buf367, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf368, primals_185, buf371, 524288, grid=grid(524288), stream=stream0)
        del primals_185
        buf372 = reinterpret_tensor(buf368, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf369, primals_187, buf372, 524288, grid=grid(524288), stream=stream0)
        del primals_187
        # Source Nodes: [], Original ATen: []
        buf373 = aten._scaled_dot_product_efficient_attention(buf370, buf371, buf372, None, True, 0.1, scale=0.125)
        buf374 = buf373[0]
        buf375 = buf373[1]
        buf376 = buf373[2]
        buf377 = buf373[3]
        del buf373
        buf378 = buf369; del buf369  # reuse
        # Source Nodes: [hidden_states_77], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf374, buf378, 524288, grid=grid(524288), stream=stream0)
        buf379 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf378, reinterpret_tensor(primals_188, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf379)
        del primals_189
        # Source Nodes: [hidden_states_78], Original ATen: [aten.native_dropout]
        buf380 = aten.native_dropout(reinterpret_tensor(buf379, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf381 = buf380[0]
        buf382 = buf380[1]
        del buf380
        buf386 = reinterpret_tensor(buf379, (1, 512, 1024), (524288, 1024, 1), 0); del buf379  # reuse
        buf387 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf830 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_20, attention_output_22, hidden_states_76, hidden_states_79, ln_output_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf329, buf349, buf360, buf381, primals_190, primals_191, buf386, buf387, buf830, 512, 1024, grid=grid(512), stream=stream0)
        del primals_191
        buf388 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, buf387, reinterpret_tensor(primals_192, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf388)
        del primals_193
        buf389 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf388, buf389, 2097152, grid=grid(2097152), stream=stream0)
        buf390 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_195, buf389, reinterpret_tensor(primals_194, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf390)
        del primals_195
        # Source Nodes: [hidden_states_82], Original ATen: [aten.native_dropout]
        buf391 = aten.native_dropout(reinterpret_tensor(buf390, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf392 = buf391[0]
        buf393 = buf391[1]
        del buf391
        buf394 = buf392; del buf392  # reuse
        buf398 = reinterpret_tensor(buf390, (1, 512, 1024), (524288, 1024, 1), 0); del buf390  # reuse
        buf399 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf829 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_20, attention_output_22, hidden_states_76, hidden_states_83, ln_outputs_12, mixed_query_layer_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf394, buf329, buf349, buf360, buf381, primals_196, primals_197, buf398, buf399, buf829, 512, 1024, grid=grid(512), stream=stream0)
        del primals_197
        buf400 = reinterpret_tensor(buf381, (512, 1024), (1024, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf399, reinterpret_tensor(primals_198, (1024, 1024), (1, 1024), 0), out=buf400)
        buf401 = reinterpret_tensor(buf360, (512, 1024), (1024, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf399, reinterpret_tensor(primals_200, (1024, 1024), (1, 1024), 0), out=buf401)
        buf402 = reinterpret_tensor(buf349, (512, 1024), (1024, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf399, reinterpret_tensor(primals_202, (1024, 1024), (1, 1024), 0), out=buf402)
        buf403 = reinterpret_tensor(buf329, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf400, primals_199, buf403, 524288, grid=grid(524288), stream=stream0)
        del primals_199
        buf404 = reinterpret_tensor(buf400, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf401, primals_201, buf404, 524288, grid=grid(524288), stream=stream0)
        del primals_201
        buf405 = reinterpret_tensor(buf401, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf402, primals_203, buf405, 524288, grid=grid(524288), stream=stream0)
        del primals_203
        # Source Nodes: [], Original ATen: []
        buf406 = aten._scaled_dot_product_efficient_attention(buf403, buf404, buf405, None, True, 0.1, scale=0.125)
        buf407 = buf406[0]
        buf408 = buf406[1]
        buf409 = buf406[2]
        buf410 = buf406[3]
        del buf406
        buf411 = buf402; del buf402  # reuse
        # Source Nodes: [hidden_states_84], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf407, buf411, 524288, grid=grid(524288), stream=stream0)
        buf412 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_205, buf411, reinterpret_tensor(primals_204, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf412)
        del primals_205
        # Source Nodes: [hidden_states_85], Original ATen: [aten.native_dropout]
        buf413 = aten.native_dropout(reinterpret_tensor(buf412, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf414 = buf413[0]
        buf415 = buf413[1]
        del buf413
        buf419 = reinterpret_tensor(buf412, (1, 512, 1024), (524288, 1024, 1), 0); del buf412  # reuse
        buf420 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf828 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_24, hidden_states_86, ln_output_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf394, buf414, primals_206, primals_207, buf419, buf420, buf828, 512, 1024, grid=grid(512), stream=stream0)
        del primals_207
        buf421 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_209, buf420, reinterpret_tensor(primals_208, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf421)
        del primals_209
        buf422 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_88, intermediate_output_12], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf421, buf422, 2097152, grid=grid(2097152), stream=stream0)
        buf423 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_211, buf422, reinterpret_tensor(primals_210, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf423)
        del primals_211
        # Source Nodes: [hidden_states_89], Original ATen: [aten.native_dropout]
        buf424 = aten.native_dropout(reinterpret_tensor(buf423, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf425 = buf424[0]
        buf426 = buf424[1]
        del buf424
        buf430 = reinterpret_tensor(buf423, (1, 512, 1024), (524288, 1024, 1), 0); del buf423  # reuse
        buf431 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf827 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_24, hidden_states_90, ln_outputs_13, mixed_query_layer_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf394, buf414, buf425, primals_212, primals_213, buf430, buf431, buf827, 512, 1024, grid=grid(512), stream=stream0)
        del primals_213
        buf432 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf431, reinterpret_tensor(primals_214, (1024, 1024), (1, 1024), 0), out=buf432)
        buf433 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf431, reinterpret_tensor(primals_216, (1024, 1024), (1, 1024), 0), out=buf433)
        buf434 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf431, reinterpret_tensor(primals_218, (1024, 1024), (1, 1024), 0), out=buf434)
        buf435 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf432, primals_215, buf435, 524288, grid=grid(524288), stream=stream0)
        del primals_215
        buf436 = reinterpret_tensor(buf432, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf433, primals_217, buf436, 524288, grid=grid(524288), stream=stream0)
        del primals_217
        buf437 = reinterpret_tensor(buf433, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf434, primals_219, buf437, 524288, grid=grid(524288), stream=stream0)
        del primals_219
        # Source Nodes: [], Original ATen: []
        buf438 = aten._scaled_dot_product_efficient_attention(buf435, buf436, buf437, None, True, 0.1, scale=0.125)
        buf439 = buf438[0]
        buf440 = buf438[1]
        buf441 = buf438[2]
        buf442 = buf438[3]
        del buf438
        buf443 = buf434; del buf434  # reuse
        # Source Nodes: [hidden_states_91], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf439, buf443, 524288, grid=grid(524288), stream=stream0)
        buf444 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_221, buf443, reinterpret_tensor(primals_220, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf444)
        del primals_221
        # Source Nodes: [hidden_states_92], Original ATen: [aten.native_dropout]
        buf445 = aten.native_dropout(reinterpret_tensor(buf444, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf446 = buf445[0]
        buf447 = buf445[1]
        del buf445
        buf451 = reinterpret_tensor(buf444, (1, 512, 1024), (524288, 1024, 1), 0); del buf444  # reuse
        buf452 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf826 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_24, attention_output_26, hidden_states_90, hidden_states_93, ln_output_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf394, buf414, buf425, buf446, primals_222, primals_223, buf451, buf452, buf826, 512, 1024, grid=grid(512), stream=stream0)
        del primals_223
        buf453 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_225, buf452, reinterpret_tensor(primals_224, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf453)
        del primals_225
        buf454 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, intermediate_output_13], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf453, buf454, 2097152, grid=grid(2097152), stream=stream0)
        buf455 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_227, buf454, reinterpret_tensor(primals_226, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf455)
        del primals_227
        # Source Nodes: [hidden_states_96], Original ATen: [aten.native_dropout]
        buf456 = aten.native_dropout(reinterpret_tensor(buf455, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf457 = buf456[0]
        buf458 = buf456[1]
        del buf456
        buf459 = buf457; del buf457  # reuse
        buf463 = reinterpret_tensor(buf455, (1, 512, 1024), (524288, 1024, 1), 0); del buf455  # reuse
        buf464 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf825 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_24, attention_output_26, hidden_states_90, hidden_states_97, ln_outputs_14, mixed_query_layer_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf459, buf394, buf414, buf425, buf446, primals_228, primals_229, buf463, buf464, buf825, 512, 1024, grid=grid(512), stream=stream0)
        del primals_229
        buf465 = reinterpret_tensor(buf446, (512, 1024), (1024, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf464, reinterpret_tensor(primals_230, (1024, 1024), (1, 1024), 0), out=buf465)
        buf466 = reinterpret_tensor(buf425, (512, 1024), (1024, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf464, reinterpret_tensor(primals_232, (1024, 1024), (1, 1024), 0), out=buf466)
        buf467 = reinterpret_tensor(buf414, (512, 1024), (1024, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf464, reinterpret_tensor(primals_234, (1024, 1024), (1, 1024), 0), out=buf467)
        buf468 = reinterpret_tensor(buf394, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf465, primals_231, buf468, 524288, grid=grid(524288), stream=stream0)
        del primals_231
        buf469 = reinterpret_tensor(buf465, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf466, primals_233, buf469, 524288, grid=grid(524288), stream=stream0)
        del primals_233
        buf470 = reinterpret_tensor(buf466, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf467, primals_235, buf470, 524288, grid=grid(524288), stream=stream0)
        del primals_235
        # Source Nodes: [], Original ATen: []
        buf471 = aten._scaled_dot_product_efficient_attention(buf468, buf469, buf470, None, True, 0.1, scale=0.125)
        buf472 = buf471[0]
        buf473 = buf471[1]
        buf474 = buf471[2]
        buf475 = buf471[3]
        del buf471
        buf476 = buf467; del buf467  # reuse
        # Source Nodes: [hidden_states_98], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf472, buf476, 524288, grid=grid(524288), stream=stream0)
        buf477 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_237, buf476, reinterpret_tensor(primals_236, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf477)
        del primals_237
        # Source Nodes: [hidden_states_99], Original ATen: [aten.native_dropout]
        buf478 = aten.native_dropout(reinterpret_tensor(buf477, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf479 = buf478[0]
        buf480 = buf478[1]
        del buf478
        buf484 = reinterpret_tensor(buf477, (1, 512, 1024), (524288, 1024, 1), 0); del buf477  # reuse
        buf485 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf824 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_28, hidden_states_100, ln_output_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf459, buf479, primals_238, primals_239, buf484, buf485, buf824, 512, 1024, grid=grid(512), stream=stream0)
        del primals_239
        buf486 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_100], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_241, buf485, reinterpret_tensor(primals_240, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf486)
        del primals_241
        buf487 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102, intermediate_output_14], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf486, buf487, 2097152, grid=grid(2097152), stream=stream0)
        buf488 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_243, buf487, reinterpret_tensor(primals_242, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf488)
        del primals_243
        # Source Nodes: [hidden_states_103], Original ATen: [aten.native_dropout]
        buf489 = aten.native_dropout(reinterpret_tensor(buf488, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf490 = buf489[0]
        buf491 = buf489[1]
        del buf489
        buf495 = reinterpret_tensor(buf488, (1, 512, 1024), (524288, 1024, 1), 0); del buf488  # reuse
        buf496 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf823 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_28, hidden_states_104, ln_outputs_15, mixed_query_layer_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf459, buf479, buf490, primals_244, primals_245, buf495, buf496, buf823, 512, 1024, grid=grid(512), stream=stream0)
        del primals_245
        buf497 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf496, reinterpret_tensor(primals_246, (1024, 1024), (1, 1024), 0), out=buf497)
        buf498 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf496, reinterpret_tensor(primals_248, (1024, 1024), (1, 1024), 0), out=buf498)
        buf499 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf496, reinterpret_tensor(primals_250, (1024, 1024), (1, 1024), 0), out=buf499)
        buf500 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf497, primals_247, buf500, 524288, grid=grid(524288), stream=stream0)
        del primals_247
        buf501 = reinterpret_tensor(buf497, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf498, primals_249, buf501, 524288, grid=grid(524288), stream=stream0)
        del primals_249
        buf502 = reinterpret_tensor(buf498, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf499, primals_251, buf502, 524288, grid=grid(524288), stream=stream0)
        del primals_251
        # Source Nodes: [], Original ATen: []
        buf503 = aten._scaled_dot_product_efficient_attention(buf500, buf501, buf502, None, True, 0.1, scale=0.125)
        buf504 = buf503[0]
        buf505 = buf503[1]
        buf506 = buf503[2]
        buf507 = buf503[3]
        del buf503
        buf508 = buf499; del buf499  # reuse
        # Source Nodes: [hidden_states_105], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf504, buf508, 524288, grid=grid(524288), stream=stream0)
        buf509 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_253, buf508, reinterpret_tensor(primals_252, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf509)
        del primals_253
        # Source Nodes: [hidden_states_106], Original ATen: [aten.native_dropout]
        buf510 = aten.native_dropout(reinterpret_tensor(buf509, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf511 = buf510[0]
        buf512 = buf510[1]
        del buf510
        buf516 = reinterpret_tensor(buf509, (1, 512, 1024), (524288, 1024, 1), 0); del buf509  # reuse
        buf517 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf822 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_28, attention_output_30, hidden_states_104, hidden_states_107, ln_output_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf459, buf479, buf490, buf511, primals_254, primals_255, buf516, buf517, buf822, 512, 1024, grid=grid(512), stream=stream0)
        del primals_255
        buf518 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_107], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_257, buf517, reinterpret_tensor(primals_256, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf518)
        del primals_257
        buf519 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, intermediate_output_15], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf518, buf519, 2097152, grid=grid(2097152), stream=stream0)
        buf520 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_259, buf519, reinterpret_tensor(primals_258, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf520)
        del primals_259
        # Source Nodes: [hidden_states_110], Original ATen: [aten.native_dropout]
        buf521 = aten.native_dropout(reinterpret_tensor(buf520, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf522 = buf521[0]
        buf523 = buf521[1]
        del buf521
        buf524 = buf522; del buf522  # reuse
        buf528 = reinterpret_tensor(buf520, (1, 512, 1024), (524288, 1024, 1), 0); del buf520  # reuse
        buf529 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf821 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_28, attention_output_30, hidden_states_104, hidden_states_111, ln_outputs_16, mixed_query_layer_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf524, buf459, buf479, buf490, buf511, primals_260, primals_261, buf528, buf529, buf821, 512, 1024, grid=grid(512), stream=stream0)
        del primals_261
        buf530 = reinterpret_tensor(buf511, (512, 1024), (1024, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf529, reinterpret_tensor(primals_262, (1024, 1024), (1, 1024), 0), out=buf530)
        buf531 = reinterpret_tensor(buf490, (512, 1024), (1024, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf529, reinterpret_tensor(primals_264, (1024, 1024), (1, 1024), 0), out=buf531)
        buf532 = reinterpret_tensor(buf479, (512, 1024), (1024, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf529, reinterpret_tensor(primals_266, (1024, 1024), (1, 1024), 0), out=buf532)
        buf533 = reinterpret_tensor(buf459, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf530, primals_263, buf533, 524288, grid=grid(524288), stream=stream0)
        del primals_263
        buf534 = reinterpret_tensor(buf530, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf531, primals_265, buf534, 524288, grid=grid(524288), stream=stream0)
        del primals_265
        buf535 = reinterpret_tensor(buf531, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf532, primals_267, buf535, 524288, grid=grid(524288), stream=stream0)
        del primals_267
        # Source Nodes: [], Original ATen: []
        buf536 = aten._scaled_dot_product_efficient_attention(buf533, buf534, buf535, None, True, 0.1, scale=0.125)
        buf537 = buf536[0]
        buf538 = buf536[1]
        buf539 = buf536[2]
        buf540 = buf536[3]
        del buf536
        buf541 = buf532; del buf532  # reuse
        # Source Nodes: [hidden_states_112], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf537, buf541, 524288, grid=grid(524288), stream=stream0)
        buf542 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_112], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_269, buf541, reinterpret_tensor(primals_268, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf542)
        del primals_269
        # Source Nodes: [hidden_states_113], Original ATen: [aten.native_dropout]
        buf543 = aten.native_dropout(reinterpret_tensor(buf542, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf544 = buf543[0]
        buf545 = buf543[1]
        del buf543
        buf549 = reinterpret_tensor(buf542, (1, 512, 1024), (524288, 1024, 1), 0); del buf542  # reuse
        buf550 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf820 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_32, hidden_states_114, ln_output_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf524, buf544, primals_270, primals_271, buf549, buf550, buf820, 512, 1024, grid=grid(512), stream=stream0)
        del primals_271
        buf551 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_273, buf550, reinterpret_tensor(primals_272, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf551)
        del primals_273
        buf552 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_116, intermediate_output_16], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf551, buf552, 2097152, grid=grid(2097152), stream=stream0)
        buf553 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_116], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_275, buf552, reinterpret_tensor(primals_274, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf553)
        del primals_275
        # Source Nodes: [hidden_states_117], Original ATen: [aten.native_dropout]
        buf554 = aten.native_dropout(reinterpret_tensor(buf553, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf555 = buf554[0]
        buf556 = buf554[1]
        del buf554
        buf560 = reinterpret_tensor(buf553, (1, 512, 1024), (524288, 1024, 1), 0); del buf553  # reuse
        buf561 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf819 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_32, hidden_states_118, ln_outputs_17, mixed_query_layer_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf524, buf544, buf555, primals_276, primals_277, buf560, buf561, buf819, 512, 1024, grid=grid(512), stream=stream0)
        del primals_277
        buf562 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf561, reinterpret_tensor(primals_278, (1024, 1024), (1, 1024), 0), out=buf562)
        buf563 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf561, reinterpret_tensor(primals_280, (1024, 1024), (1, 1024), 0), out=buf563)
        buf564 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf561, reinterpret_tensor(primals_282, (1024, 1024), (1, 1024), 0), out=buf564)
        buf565 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf562, primals_279, buf565, 524288, grid=grid(524288), stream=stream0)
        del primals_279
        buf566 = reinterpret_tensor(buf562, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf563, primals_281, buf566, 524288, grid=grid(524288), stream=stream0)
        del primals_281
        buf567 = reinterpret_tensor(buf563, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf564, primals_283, buf567, 524288, grid=grid(524288), stream=stream0)
        del primals_283
        # Source Nodes: [], Original ATen: []
        buf568 = aten._scaled_dot_product_efficient_attention(buf565, buf566, buf567, None, True, 0.1, scale=0.125)
        buf569 = buf568[0]
        buf570 = buf568[1]
        buf571 = buf568[2]
        buf572 = buf568[3]
        del buf568
        buf573 = buf564; del buf564  # reuse
        # Source Nodes: [hidden_states_119], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf569, buf573, 524288, grid=grid(524288), stream=stream0)
        buf574 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_119], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_285, buf573, reinterpret_tensor(primals_284, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf574)
        del primals_285
        # Source Nodes: [hidden_states_120], Original ATen: [aten.native_dropout]
        buf575 = aten.native_dropout(reinterpret_tensor(buf574, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf576 = buf575[0]
        buf577 = buf575[1]
        del buf575
        buf581 = reinterpret_tensor(buf574, (1, 512, 1024), (524288, 1024, 1), 0); del buf574  # reuse
        buf582 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf818 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_32, attention_output_34, hidden_states_118, hidden_states_121, ln_output_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf524, buf544, buf555, buf576, primals_286, primals_287, buf581, buf582, buf818, 512, 1024, grid=grid(512), stream=stream0)
        del primals_287
        buf583 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_121], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_289, buf582, reinterpret_tensor(primals_288, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf583)
        del primals_289
        buf584 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_123, intermediate_output_17], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf583, buf584, 2097152, grid=grid(2097152), stream=stream0)
        buf585 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_291, buf584, reinterpret_tensor(primals_290, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf585)
        del primals_291
        # Source Nodes: [hidden_states_124], Original ATen: [aten.native_dropout]
        buf586 = aten.native_dropout(reinterpret_tensor(buf585, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf587 = buf586[0]
        buf588 = buf586[1]
        del buf586
        buf589 = buf587; del buf587  # reuse
        buf593 = reinterpret_tensor(buf585, (1, 512, 1024), (524288, 1024, 1), 0); del buf585  # reuse
        buf594 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf817 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_32, attention_output_34, hidden_states_118, hidden_states_125, ln_outputs_18, mixed_query_layer_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf589, buf524, buf544, buf555, buf576, primals_292, primals_293, buf593, buf594, buf817, 512, 1024, grid=grid(512), stream=stream0)
        del primals_293
        buf595 = reinterpret_tensor(buf576, (512, 1024), (1024, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf594, reinterpret_tensor(primals_294, (1024, 1024), (1, 1024), 0), out=buf595)
        buf596 = reinterpret_tensor(buf555, (512, 1024), (1024, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf594, reinterpret_tensor(primals_296, (1024, 1024), (1, 1024), 0), out=buf596)
        buf597 = reinterpret_tensor(buf544, (512, 1024), (1024, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf594, reinterpret_tensor(primals_298, (1024, 1024), (1, 1024), 0), out=buf597)
        buf598 = reinterpret_tensor(buf524, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf595, primals_295, buf598, 524288, grid=grid(524288), stream=stream0)
        del primals_295
        buf599 = reinterpret_tensor(buf595, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf595  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf596, primals_297, buf599, 524288, grid=grid(524288), stream=stream0)
        del primals_297
        buf600 = reinterpret_tensor(buf596, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf597, primals_299, buf600, 524288, grid=grid(524288), stream=stream0)
        del primals_299
        # Source Nodes: [], Original ATen: []
        buf601 = aten._scaled_dot_product_efficient_attention(buf598, buf599, buf600, None, True, 0.1, scale=0.125)
        buf602 = buf601[0]
        buf603 = buf601[1]
        buf604 = buf601[2]
        buf605 = buf601[3]
        del buf601
        buf606 = buf597; del buf597  # reuse
        # Source Nodes: [hidden_states_126], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf602, buf606, 524288, grid=grid(524288), stream=stream0)
        buf607 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_126], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_301, buf606, reinterpret_tensor(primals_300, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf607)
        del primals_301
        # Source Nodes: [hidden_states_127], Original ATen: [aten.native_dropout]
        buf608 = aten.native_dropout(reinterpret_tensor(buf607, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf609 = buf608[0]
        buf610 = buf608[1]
        del buf608
        buf614 = reinterpret_tensor(buf607, (1, 512, 1024), (524288, 1024, 1), 0); del buf607  # reuse
        buf615 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf816 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_36, hidden_states_128, ln_output_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf589, buf609, primals_302, primals_303, buf614, buf615, buf816, 512, 1024, grid=grid(512), stream=stream0)
        del primals_303
        buf616 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_128], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_305, buf615, reinterpret_tensor(primals_304, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf616)
        del primals_305
        buf617 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_130, intermediate_output_18], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf616, buf617, 2097152, grid=grid(2097152), stream=stream0)
        buf618 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_130], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_307, buf617, reinterpret_tensor(primals_306, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf618)
        del primals_307
        # Source Nodes: [hidden_states_131], Original ATen: [aten.native_dropout]
        buf619 = aten.native_dropout(reinterpret_tensor(buf618, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf620 = buf619[0]
        buf621 = buf619[1]
        del buf619
        buf625 = reinterpret_tensor(buf618, (1, 512, 1024), (524288, 1024, 1), 0); del buf618  # reuse
        buf626 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf815 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_36, hidden_states_132, ln_outputs_19, mixed_query_layer_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf589, buf609, buf620, primals_308, primals_309, buf625, buf626, buf815, 512, 1024, grid=grid(512), stream=stream0)
        del primals_309
        buf627 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf626, reinterpret_tensor(primals_310, (1024, 1024), (1, 1024), 0), out=buf627)
        buf628 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf626, reinterpret_tensor(primals_312, (1024, 1024), (1, 1024), 0), out=buf628)
        buf629 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf626, reinterpret_tensor(primals_314, (1024, 1024), (1, 1024), 0), out=buf629)
        buf630 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf627, primals_311, buf630, 524288, grid=grid(524288), stream=stream0)
        del primals_311
        buf631 = reinterpret_tensor(buf627, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf627  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf628, primals_313, buf631, 524288, grid=grid(524288), stream=stream0)
        del primals_313
        buf632 = reinterpret_tensor(buf628, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf629, primals_315, buf632, 524288, grid=grid(524288), stream=stream0)
        del primals_315
        # Source Nodes: [], Original ATen: []
        buf633 = aten._scaled_dot_product_efficient_attention(buf630, buf631, buf632, None, True, 0.1, scale=0.125)
        buf634 = buf633[0]
        buf635 = buf633[1]
        buf636 = buf633[2]
        buf637 = buf633[3]
        del buf633
        buf638 = buf629; del buf629  # reuse
        # Source Nodes: [hidden_states_133], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf634, buf638, 524288, grid=grid(524288), stream=stream0)
        buf639 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_317, buf638, reinterpret_tensor(primals_316, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf639)
        del primals_317
        # Source Nodes: [hidden_states_134], Original ATen: [aten.native_dropout]
        buf640 = aten.native_dropout(reinterpret_tensor(buf639, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf641 = buf640[0]
        buf642 = buf640[1]
        del buf640
        buf646 = reinterpret_tensor(buf639, (1, 512, 1024), (524288, 1024, 1), 0); del buf639  # reuse
        buf647 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf814 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_36, attention_output_38, hidden_states_132, hidden_states_135, ln_output_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf589, buf609, buf620, buf641, primals_318, primals_319, buf646, buf647, buf814, 512, 1024, grid=grid(512), stream=stream0)
        del primals_319
        buf648 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_135], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_321, buf647, reinterpret_tensor(primals_320, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf648)
        del primals_321
        buf649 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_137, intermediate_output_19], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf648, buf649, 2097152, grid=grid(2097152), stream=stream0)
        buf650 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_137], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_323, buf649, reinterpret_tensor(primals_322, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf650)
        del primals_323
        # Source Nodes: [hidden_states_138], Original ATen: [aten.native_dropout]
        buf651 = aten.native_dropout(reinterpret_tensor(buf650, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf652 = buf651[0]
        buf653 = buf651[1]
        del buf651
        buf654 = buf652; del buf652  # reuse
        buf658 = reinterpret_tensor(buf650, (1, 512, 1024), (524288, 1024, 1), 0); del buf650  # reuse
        buf659 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf813 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_36, attention_output_38, hidden_states_132, hidden_states_139, ln_outputs_20, mixed_query_layer_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf654, buf589, buf609, buf620, buf641, primals_324, primals_325, buf658, buf659, buf813, 512, 1024, grid=grid(512), stream=stream0)
        del primals_325
        buf660 = reinterpret_tensor(buf641, (512, 1024), (1024, 1), 0); del buf641  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf659, reinterpret_tensor(primals_326, (1024, 1024), (1, 1024), 0), out=buf660)
        buf661 = reinterpret_tensor(buf620, (512, 1024), (1024, 1), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf659, reinterpret_tensor(primals_328, (1024, 1024), (1, 1024), 0), out=buf661)
        buf662 = reinterpret_tensor(buf609, (512, 1024), (1024, 1), 0); del buf609  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf659, reinterpret_tensor(primals_330, (1024, 1024), (1, 1024), 0), out=buf662)
        buf663 = reinterpret_tensor(buf589, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf589  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf660, primals_327, buf663, 524288, grid=grid(524288), stream=stream0)
        del primals_327
        buf664 = reinterpret_tensor(buf660, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf661, primals_329, buf664, 524288, grid=grid(524288), stream=stream0)
        del primals_329
        buf665 = reinterpret_tensor(buf661, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf661  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf662, primals_331, buf665, 524288, grid=grid(524288), stream=stream0)
        del primals_331
        # Source Nodes: [], Original ATen: []
        buf666 = aten._scaled_dot_product_efficient_attention(buf663, buf664, buf665, None, True, 0.1, scale=0.125)
        buf667 = buf666[0]
        buf668 = buf666[1]
        buf669 = buf666[2]
        buf670 = buf666[3]
        del buf666
        buf671 = buf662; del buf662  # reuse
        # Source Nodes: [hidden_states_140], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf667, buf671, 524288, grid=grid(524288), stream=stream0)
        buf672 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_140], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_333, buf671, reinterpret_tensor(primals_332, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf672)
        del primals_333
        # Source Nodes: [hidden_states_141], Original ATen: [aten.native_dropout]
        buf673 = aten.native_dropout(reinterpret_tensor(buf672, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf674 = buf673[0]
        buf675 = buf673[1]
        del buf673
        buf679 = reinterpret_tensor(buf672, (1, 512, 1024), (524288, 1024, 1), 0); del buf672  # reuse
        buf680 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf812 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_40, hidden_states_142, ln_output_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf654, buf674, primals_334, primals_335, buf679, buf680, buf812, 512, 1024, grid=grid(512), stream=stream0)
        del primals_335
        buf681 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_142], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_337, buf680, reinterpret_tensor(primals_336, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf681)
        del primals_337
        buf682 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_144, intermediate_output_20], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf681, buf682, 2097152, grid=grid(2097152), stream=stream0)
        buf683 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_339, buf682, reinterpret_tensor(primals_338, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf683)
        del primals_339
        # Source Nodes: [hidden_states_145], Original ATen: [aten.native_dropout]
        buf684 = aten.native_dropout(reinterpret_tensor(buf683, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf685 = buf684[0]
        buf686 = buf684[1]
        del buf684
        buf690 = reinterpret_tensor(buf683, (1, 512, 1024), (524288, 1024, 1), 0); del buf683  # reuse
        buf691 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf811 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_40, hidden_states_146, ln_outputs_21, mixed_query_layer_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf654, buf674, buf685, primals_340, primals_341, buf690, buf691, buf811, 512, 1024, grid=grid(512), stream=stream0)
        del primals_341
        buf692 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf691, reinterpret_tensor(primals_342, (1024, 1024), (1, 1024), 0), out=buf692)
        buf693 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf691, reinterpret_tensor(primals_344, (1024, 1024), (1, 1024), 0), out=buf693)
        buf694 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf691, reinterpret_tensor(primals_346, (1024, 1024), (1, 1024), 0), out=buf694)
        buf695 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf692, primals_343, buf695, 524288, grid=grid(524288), stream=stream0)
        del primals_343
        buf696 = reinterpret_tensor(buf692, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf693, primals_345, buf696, 524288, grid=grid(524288), stream=stream0)
        del primals_345
        buf697 = reinterpret_tensor(buf693, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf693  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf694, primals_347, buf697, 524288, grid=grid(524288), stream=stream0)
        del primals_347
        # Source Nodes: [], Original ATen: []
        buf698 = aten._scaled_dot_product_efficient_attention(buf695, buf696, buf697, None, True, 0.1, scale=0.125)
        buf699 = buf698[0]
        buf700 = buf698[1]
        buf701 = buf698[2]
        buf702 = buf698[3]
        del buf698
        buf703 = buf694; del buf694  # reuse
        # Source Nodes: [hidden_states_147], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf699, buf703, 524288, grid=grid(524288), stream=stream0)
        buf704 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_147], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_349, buf703, reinterpret_tensor(primals_348, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf704)
        del primals_349
        # Source Nodes: [hidden_states_148], Original ATen: [aten.native_dropout]
        buf705 = aten.native_dropout(reinterpret_tensor(buf704, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf706 = buf705[0]
        buf707 = buf705[1]
        del buf705
        buf711 = reinterpret_tensor(buf704, (1, 512, 1024), (524288, 1024, 1), 0); del buf704  # reuse
        buf712 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf810 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_40, attention_output_42, hidden_states_146, hidden_states_149, ln_output_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf654, buf674, buf685, buf706, primals_350, primals_351, buf711, buf712, buf810, 512, 1024, grid=grid(512), stream=stream0)
        del primals_351
        buf713 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_149], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_353, buf712, reinterpret_tensor(primals_352, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf713)
        del primals_353
        buf714 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_151, intermediate_output_21], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf713, buf714, 2097152, grid=grid(2097152), stream=stream0)
        buf715 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_151], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_355, buf714, reinterpret_tensor(primals_354, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf715)
        del primals_355
        # Source Nodes: [hidden_states_152], Original ATen: [aten.native_dropout]
        buf716 = aten.native_dropout(reinterpret_tensor(buf715, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf717 = buf716[0]
        buf718 = buf716[1]
        del buf716
        buf719 = buf717; del buf717  # reuse
        buf723 = reinterpret_tensor(buf715, (1, 512, 1024), (524288, 1024, 1), 0); del buf715  # reuse
        buf724 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf809 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_40, attention_output_42, hidden_states_146, hidden_states_153, ln_outputs_22, mixed_query_layer_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf719, buf654, buf674, buf685, buf706, primals_356, primals_357, buf723, buf724, buf809, 512, 1024, grid=grid(512), stream=stream0)
        del primals_357
        buf725 = reinterpret_tensor(buf706, (512, 1024), (1024, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf724, reinterpret_tensor(primals_358, (1024, 1024), (1, 1024), 0), out=buf725)
        buf726 = reinterpret_tensor(buf685, (512, 1024), (1024, 1), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf724, reinterpret_tensor(primals_360, (1024, 1024), (1, 1024), 0), out=buf726)
        buf727 = reinterpret_tensor(buf674, (512, 1024), (1024, 1), 0); del buf674  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf724, reinterpret_tensor(primals_362, (1024, 1024), (1, 1024), 0), out=buf727)
        buf728 = reinterpret_tensor(buf654, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf654  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf725, primals_359, buf728, 524288, grid=grid(524288), stream=stream0)
        del primals_359
        buf729 = reinterpret_tensor(buf725, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf726, primals_361, buf729, 524288, grid=grid(524288), stream=stream0)
        del primals_361
        buf730 = reinterpret_tensor(buf726, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf726  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf727, primals_363, buf730, 524288, grid=grid(524288), stream=stream0)
        del primals_363
        # Source Nodes: [], Original ATen: []
        buf731 = aten._scaled_dot_product_efficient_attention(buf728, buf729, buf730, None, True, 0.1, scale=0.125)
        buf732 = buf731[0]
        buf733 = buf731[1]
        buf734 = buf731[2]
        buf735 = buf731[3]
        del buf731
        buf736 = buf727; del buf727  # reuse
        # Source Nodes: [hidden_states_154], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf732, buf736, 524288, grid=grid(524288), stream=stream0)
        buf737 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_154], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_365, buf736, reinterpret_tensor(primals_364, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf737)
        del primals_365
        # Source Nodes: [hidden_states_155], Original ATen: [aten.native_dropout]
        buf738 = aten.native_dropout(reinterpret_tensor(buf737, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf739 = buf738[0]
        buf740 = buf738[1]
        del buf738
        buf744 = reinterpret_tensor(buf737, (1, 512, 1024), (524288, 1024, 1), 0); del buf737  # reuse
        buf745 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf808 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_44, hidden_states_156, ln_output_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf719, buf739, primals_366, primals_367, buf744, buf745, buf808, 512, 1024, grid=grid(512), stream=stream0)
        del primals_367
        buf746 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_156], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_369, buf745, reinterpret_tensor(primals_368, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf746)
        del primals_369
        buf747 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_158, intermediate_output_22], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf746, buf747, 2097152, grid=grid(2097152), stream=stream0)
        buf748 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_158], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_371, buf747, reinterpret_tensor(primals_370, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf748)
        del primals_371
        # Source Nodes: [hidden_states_159], Original ATen: [aten.native_dropout]
        buf749 = aten.native_dropout(reinterpret_tensor(buf748, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf750 = buf749[0]
        buf751 = buf749[1]
        del buf749
        buf755 = reinterpret_tensor(buf748, (1, 512, 1024), (524288, 1024, 1), 0); del buf748  # reuse
        buf756 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf807 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_44, hidden_states_160, ln_outputs_23, mixed_query_layer_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf719, buf739, buf750, primals_372, primals_373, buf755, buf756, buf807, 512, 1024, grid=grid(512), stream=stream0)
        del primals_373
        buf757 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf756, reinterpret_tensor(primals_374, (1024, 1024), (1, 1024), 0), out=buf757)
        buf758 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf756, reinterpret_tensor(primals_376, (1024, 1024), (1, 1024), 0), out=buf758)
        buf759 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf756, reinterpret_tensor(primals_378, (1024, 1024), (1, 1024), 0), out=buf759)
        buf760 = empty((1, 16, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf757, primals_375, buf760, 524288, grid=grid(524288), stream=stream0)
        del primals_375
        buf761 = reinterpret_tensor(buf757, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf757  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf758, primals_377, buf761, 524288, grid=grid(524288), stream=stream0)
        del primals_377
        buf762 = reinterpret_tensor(buf758, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf758  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf759, primals_379, buf762, 524288, grid=grid(524288), stream=stream0)
        del primals_379
        # Source Nodes: [], Original ATen: []
        buf763 = aten._scaled_dot_product_efficient_attention(buf760, buf761, buf762, None, True, 0.1, scale=0.125)
        buf764 = buf763[0]
        buf765 = buf763[1]
        buf766 = buf763[2]
        buf767 = buf763[3]
        del buf763
        buf768 = buf759; del buf759  # reuse
        # Source Nodes: [hidden_states_161], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf764, buf768, 524288, grid=grid(524288), stream=stream0)
        buf769 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_161], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_381, buf768, reinterpret_tensor(primals_380, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf769)
        del primals_381
        # Source Nodes: [hidden_states_162], Original ATen: [aten.native_dropout]
        buf770 = aten.native_dropout(reinterpret_tensor(buf769, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf771 = buf770[0]
        buf772 = buf770[1]
        del buf770
        buf776 = reinterpret_tensor(buf769, (1, 512, 1024), (524288, 1024, 1), 0); del buf769  # reuse
        buf777 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf806 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_44, attention_output_46, hidden_states_160, hidden_states_163, ln_output_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf719, buf739, buf750, buf771, primals_382, primals_383, buf776, buf777, buf806, 512, 1024, grid=grid(512), stream=stream0)
        del primals_383
        buf778 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_385, buf777, reinterpret_tensor(primals_384, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf778)
        del primals_385
        buf779 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_165, intermediate_output_23], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf778, buf779, 2097152, grid=grid(2097152), stream=stream0)
        buf780 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_387, buf779, reinterpret_tensor(primals_386, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf780)
        del primals_387
        # Source Nodes: [hidden_states_166], Original ATen: [aten.native_dropout]
        buf781 = aten.native_dropout(reinterpret_tensor(buf780, (1, 512, 1024), (524288, 1024, 1), 0), 0.1, True)
        buf782 = buf781[0]
        buf783 = buf781[1]
        del buf781
        buf784 = buf782; del buf782  # reuse
        buf788 = reinterpret_tensor(buf780, (1, 512, 1024), (524288, 1024, 1), 0); del buf780  # reuse
        buf789 = empty((512, 1024), device='cuda', dtype=torch.float32)
        buf805 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_output_44, attention_output_46, hidden_states_160, hidden_states_167, logits, sequence_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf784, buf719, buf739, buf750, buf771, primals_388, primals_389, buf788, buf789, buf805, 512, 1024, grid=grid(512), stream=stream0)
        del buf719
        del buf739
        del buf750
        del buf771
        del buf784
        del primals_389
        buf790 = empty((512, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf789, reinterpret_tensor(primals_390, (1024, 2), (1, 1024), 0), out=buf790)
        buf791 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf795 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_10.run(buf790, primals_391, buf791, buf795, 1, 512, grid=grid(1), stream=stream0)
        buf792 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf799 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_11.run(buf790, primals_391, buf792, buf799, 1, 512, grid=grid(1), stream=stream0)
        del buf790
        del primals_391
        buf796 = empty((1, ), device='cuda', dtype=torch.bool)
        buf800 = empty((1, ), device='cuda', dtype=torch.bool)
        buf854 = empty((), device='cuda', dtype=torch.float32)
        buf801 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf802 = empty((1, 1), device='cuda', dtype=torch.int64)
        buf803 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf804 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [add_73, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_12.run(primals_394, primals_395, buf795, buf799, buf796, buf800, buf854, buf801, buf802, buf803, buf804, 1, grid=grid(1), stream=stream0)
        del primals_394
        del primals_395
        return (buf854, buf791, buf792, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_393, buf0, primals_392, buf4, buf8, buf9, buf13, buf14, buf15, buf18, buf19, buf20, buf17, buf21, buf25, buf29, buf30, buf31, buf32, buf36, buf40, buf41, buf45, buf46, buf47, buf50, buf51, buf52, buf49, buf53, buf57, buf61, buf62, buf63, buf64, buf68, buf73, buf74, buf78, buf79, buf80, buf83, buf84, buf85, buf82, buf86, buf90, buf94, buf95, buf96, buf97, buf101, buf105, buf106, buf110, buf111, buf112, buf115, buf116, buf117, buf114, buf118, buf122, buf126, buf127, buf128, buf129, buf133, buf138, buf139, buf143, buf144, buf145, buf148, buf149, buf150, buf147, buf151, buf155, buf159, buf160, buf161, buf162, buf166, buf170, buf171, buf175, buf176, buf177, buf180, buf181, buf182, buf179, buf183, buf187, buf191, buf192, buf193, buf194, buf198, buf203, buf204, buf208, buf209, buf210, buf213, buf214, buf215, buf212, buf216, buf220, buf224, buf225, buf226, buf227, buf231, buf235, buf236, buf240, buf241, buf242, buf245, buf246, buf247, buf244, buf248, buf252, buf256, buf257, buf258, buf259, buf263, buf268, buf269, buf273, buf274, buf275, buf278, buf279, buf280, buf277, buf281, buf285, buf289, buf290, buf291, buf292, buf296, buf300, buf301, buf305, buf306, buf307, buf310, buf311, buf312, buf309, buf313, buf317, buf321, buf322, buf323, buf324, buf328, buf333, buf334, buf338, buf339, buf340, buf343, buf344, buf345, buf342, buf346, buf350, buf354, buf355, buf356, buf357, buf361, buf365, buf366, buf370, buf371, buf372, buf375, buf376, buf377, buf374, buf378, buf382, buf386, buf387, buf388, buf389, buf393, buf398, buf399, buf403, buf404, buf405, buf408, buf409, buf410, buf407, buf411, buf415, buf419, buf420, buf421, buf422, buf426, buf430, buf431, buf435, buf436, buf437, buf440, buf441, buf442, buf439, buf443, buf447, buf451, buf452, buf453, buf454, buf458, buf463, buf464, buf468, buf469, buf470, buf473, buf474, buf475, buf472, buf476, buf480, buf484, buf485, buf486, buf487, buf491, buf495, buf496, buf500, buf501, buf502, buf505, buf506, buf507, buf504, buf508, buf512, buf516, buf517, buf518, buf519, buf523, buf528, buf529, buf533, buf534, buf535, buf538, buf539, buf540, buf537, buf541, buf545, buf549, buf550, buf551, buf552, buf556, buf560, buf561, buf565, buf566, buf567, buf570, buf571, buf572, buf569, buf573, buf577, buf581, buf582, buf583, buf584, buf588, buf593, buf594, buf598, buf599, buf600, buf603, buf604, buf605, buf602, buf606, buf610, buf614, buf615, buf616, buf617, buf621, buf625, buf626, buf630, buf631, buf632, buf635, buf636, buf637, buf634, buf638, buf642, buf646, buf647, buf648, buf649, buf653, buf658, buf659, buf663, buf664, buf665, buf668, buf669, buf670, buf667, buf671, buf675, buf679, buf680, buf681, buf682, buf686, buf690, buf691, buf695, buf696, buf697, buf700, buf701, buf702, buf699, buf703, buf707, buf711, buf712, buf713, buf714, buf718, buf723, buf724, buf728, buf729, buf730, buf733, buf734, buf735, buf732, buf736, buf740, buf744, buf745, buf746, buf747, buf751, buf755, buf756, buf760, buf761, buf762, buf765, buf766, buf767, buf764, buf768, buf772, buf776, buf777, buf778, buf779, buf783, buf788, buf789, buf795, buf796, buf799, buf800, buf801, buf802, buf803, buf804, reinterpret_tensor(primals_390, (2, 1024), (1024, 1), 0), buf805, reinterpret_tensor(primals_386, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_384, (4096, 1024), (1024, 1), 0), buf806, reinterpret_tensor(primals_380, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_378, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_376, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_374, (1024, 1024), (1024, 1), 0), buf807, reinterpret_tensor(primals_370, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_368, (4096, 1024), (1024, 1), 0), buf808, reinterpret_tensor(primals_364, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_362, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_360, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_358, (1024, 1024), (1024, 1), 0), buf809, reinterpret_tensor(primals_354, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_352, (4096, 1024), (1024, 1), 0), buf810, reinterpret_tensor(primals_348, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_346, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_344, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_342, (1024, 1024), (1024, 1), 0), buf811, reinterpret_tensor(primals_338, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_336, (4096, 1024), (1024, 1), 0), buf812, reinterpret_tensor(primals_332, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_330, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_328, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_326, (1024, 1024), (1024, 1), 0), buf813, reinterpret_tensor(primals_322, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_320, (4096, 1024), (1024, 1), 0), buf814, reinterpret_tensor(primals_316, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_314, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_312, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_310, (1024, 1024), (1024, 1), 0), buf815, reinterpret_tensor(primals_306, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_304, (4096, 1024), (1024, 1), 0), buf816, reinterpret_tensor(primals_300, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_298, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_296, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_294, (1024, 1024), (1024, 1), 0), buf817, reinterpret_tensor(primals_290, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_288, (4096, 1024), (1024, 1), 0), buf818, reinterpret_tensor(primals_284, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_280, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_278, (1024, 1024), (1024, 1), 0), buf819, reinterpret_tensor(primals_274, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_272, (4096, 1024), (1024, 1), 0), buf820, reinterpret_tensor(primals_268, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_266, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_264, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_262, (1024, 1024), (1024, 1), 0), buf821, reinterpret_tensor(primals_258, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_256, (4096, 1024), (1024, 1), 0), buf822, reinterpret_tensor(primals_252, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_250, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_248, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_246, (1024, 1024), (1024, 1), 0), buf823, reinterpret_tensor(primals_242, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_240, (4096, 1024), (1024, 1), 0), buf824, reinterpret_tensor(primals_236, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_234, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_230, (1024, 1024), (1024, 1), 0), buf825, reinterpret_tensor(primals_226, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_224, (4096, 1024), (1024, 1), 0), buf826, reinterpret_tensor(primals_220, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_218, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_216, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_214, (1024, 1024), (1024, 1), 0), buf827, reinterpret_tensor(primals_210, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_208, (4096, 1024), (1024, 1), 0), buf828, reinterpret_tensor(primals_204, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_202, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_200, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_198, (1024, 1024), (1024, 1), 0), buf829, reinterpret_tensor(primals_194, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_192, (4096, 1024), (1024, 1), 0), buf830, reinterpret_tensor(primals_188, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_186, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_184, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_182, (1024, 1024), (1024, 1), 0), buf831, reinterpret_tensor(primals_178, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_176, (4096, 1024), (1024, 1), 0), buf832, reinterpret_tensor(primals_172, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_170, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_168, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_166, (1024, 1024), (1024, 1), 0), buf833, reinterpret_tensor(primals_162, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_160, (4096, 1024), (1024, 1), 0), buf834, reinterpret_tensor(primals_156, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_154, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_152, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_150, (1024, 1024), (1024, 1), 0), buf835, reinterpret_tensor(primals_146, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_144, (4096, 1024), (1024, 1), 0), buf836, reinterpret_tensor(primals_140, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_138, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_136, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_134, (1024, 1024), (1024, 1), 0), buf837, reinterpret_tensor(primals_130, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_128, (4096, 1024), (1024, 1), 0), buf838, reinterpret_tensor(primals_124, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_122, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_120, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_118, (1024, 1024), (1024, 1), 0), buf839, reinterpret_tensor(primals_114, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_112, (4096, 1024), (1024, 1), 0), buf840, reinterpret_tensor(primals_108, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_106, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_104, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_102, (1024, 1024), (1024, 1), 0), buf841, reinterpret_tensor(primals_98, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_96, (4096, 1024), (1024, 1), 0), buf842, reinterpret_tensor(primals_92, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_90, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_88, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_86, (1024, 1024), (1024, 1), 0), buf843, reinterpret_tensor(primals_82, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_80, (4096, 1024), (1024, 1), 0), buf844, reinterpret_tensor(primals_76, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_74, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_72, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_70, (1024, 1024), (1024, 1), 0), buf845, reinterpret_tensor(primals_66, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_64, (4096, 1024), (1024, 1), 0), buf846, reinterpret_tensor(primals_60, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_58, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_56, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_54, (1024, 1024), (1024, 1), 0), buf847, reinterpret_tensor(primals_50, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_48, (4096, 1024), (1024, 1), 0), buf848, reinterpret_tensor(primals_44, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_42, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_40, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_38, (1024, 1024), (1024, 1), 0), buf849, reinterpret_tensor(primals_34, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_32, (4096, 1024), (1024, 1), 0), buf850, reinterpret_tensor(primals_28, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_26, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_24, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_22, (1024, 1024), (1024, 1), 0), buf851, reinterpret_tensor(primals_18, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_16, (4096, 1024), (1024, 1), 0), buf852, reinterpret_tensor(primals_12, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_10, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_8, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_6, (1024, 1024), (1024, 1), 0), buf853, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((29056, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_393 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_394 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_395 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MegatronBertForQuestionAnswering', benchmark_compiled_module)
