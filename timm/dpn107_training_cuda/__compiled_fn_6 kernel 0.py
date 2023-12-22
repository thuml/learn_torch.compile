
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


# kernel path: /tmp/torchinductor_youkaichao/rh/crhdqbmfhqxh33jagwlqfsbltebc34zla7cz4vaq6ius7p2gw257.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdlujdlstvphi6nu6y4b36n2yjobipzaksf5uiapykrjkwxhvfq.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2688
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (131712*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2688*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (49*x0) + (131712*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqqi6bcvyiloufpejinxrjqr6rsyb3vysekxyohypgooppiycpt.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2688
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.002551020408163265
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjl7xls54q5crkhtjjxo7rfuhytuphisuxg4mytw57faqnzuyp6.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 852992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2176
    x2 = (xindex // 106624)
    x3 = xindex % 106624
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2048, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (25088 + x3 + (131712*x2)), tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x3 + (131712*x2)), tmp8 & xmask, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxixsi55pdvbm4z4yqn6y5scynbqj4azitralizt7imftf4jen2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1600
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (78400*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (78400*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (78400*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4z7nzqtnxstvbvdytcv6zkklxdwgdmo75fl3z4rr5ra2nq4a6ic.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 627200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1600
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crb6oipcyx6vduuti2xdq5fjpcfbehc4dcz622bvv2atqx22rhz6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2560
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (125440*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (125440*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (125440*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gh/cghh4uhijnbfanoh64gt42brqp3nvknxu555pwqsubteg7awomwt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2560
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz243p6dsodn52yto35nfrlzrfcg3qdmns23lnr4gu7g57so34ok.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 852992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2176
    x2 = (xindex // 106624)
    x3 = xindex % 106624
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2048, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (18816 + x3 + (131712*x2)), tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (18816 + x3 + (125440*x2)), tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x3 + (131712*x2)), tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (125440*x2)), tmp10 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4y/c4ytrgxeh6d5omoqdnfyhyj55z7c3ckfcdtybiphbca236imejjg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2432
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (119168*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (119168*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (119168*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coenssnocktj3g6irgkkc4fwktvtbgra3i6p6mlkprntidpivb5l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2432
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg67pmnhmpq6nhtydjdbv6jugznvs7p47d2w5gou45dk3zqxpone.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 852992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2176
    x2 = (xindex // 106624)
    x3 = xindex % 106624
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2048, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (12544 + x3 + (131712*x2)), tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (12544 + x3 + (125440*x2)), tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (12544 + x3 + (119168*x2)), tmp2 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x3 + (131712*x2)), tmp12 & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (125440*x2)), tmp12 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (119168*x2)), tmp12 & xmask, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp37vxvys7q5enjpxxftnvkcje7tthldzsuz2xjowvtvzgd65zkd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1600
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4wxn2csbppftgdwhjrz2id3phj5lhx2lhma72b4vagnkpi673uq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1600
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wz/cwz3uyx4heeh74klghaygcln5quqs6fwpqparxuqpr2kkw2z74ez.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2304
    x2 = (xindex // 112896)
    x3 = xindex % 112896
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2048, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + (131712*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x3 + (125440*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x3 + (119168*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x3 + (131712*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (125440*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (119168*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3d7oo4qcqamslnyrsy6cyefhw4ykog2oyzpemz62pmccjcajdr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2432
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (476672*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (476672*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (476672*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r1 + (196*x0) + (476672*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (196*x0) + (476672*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp16 = tmp15 <= tmp1
        tmp18 = tl.where(tmp16, tmp1, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp22 = tmp18 * tmp10
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp13 * tmp26
    tmp28 = tmp24 * tmp26
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr5 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpnl3nnv4xv3pfcf4uwaualphmf2t6h6h3wsjbipcpngc7haqy2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3813376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2432
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x3), None)
    tmp24 = tl.load(in_ptr8 + (x3), None)
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22 <= tmp1
    tmp25 = tl.where(tmp23, tmp1, tmp24)
    tmp27 = tmp26 * tmp9
    tmp28 = tmp27 * tmp12
    tmp29 = tmp7 * tmp28
    tmp30 = tmp25 - tmp29
    tmp32 = tmp31 * tmp9
    tmp33 = tmp30 - tmp32
    tmp35 = tmp11 * tmp34
    tmp36 = tmp33 * tmp35
    tmp37 = tmp21 + tmp36
    tl.store(in_out_ptr0 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4umiweaik2vndj5leaeonnodfpy26td6gnlnormf5wvuvdp3gm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (263424 + x3 + (476672*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x3 + (476672*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyo2f5dnuqk2fbwa33d2fjxjdvxmw22k3eghouygsby3rl3dn5b.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 800
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (156800*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (156800*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (156800*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4cnfzrgnmwv67d3uoov6ajpmvmrgcv2w5gcm63cunkbpvw472ws.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 800
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckmizhy7lqmrqoa5k5czik4ctcjlqdfjvbf2uhwdwnatbjsy25qq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2368
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (464128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (464128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (464128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3gt5kpxkh7n2ia2eychg3zna6qua4sgirvw4cg7psflfsnoqi7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3713024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2368
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqph2nyoxjjb5dpdxikmn4hgsakl32lomhgi54m6dpmi2g54ui5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (250880 + x3 + (476672*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (250880 + x3 + (464128*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x3 + (476672*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (464128*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/ctsdt6jahh5vx7yqouri4ngbxs4d535htfl4r6scjdx3zbfltada.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (451584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (451584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (451584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coip2dixvev2eov4srq53jjnvoebyaqkstb4cc6gkujot3wlt7wh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2304
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2odwhjlk7mtxtkjusx6bsnvnx4oryeed3t7y7rmzhwwhdwg2zy.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (238336 + x3 + (476672*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (238336 + x3 + (464128*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (238336 + x3 + (451584*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x3 + (476672*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (464128*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (451584*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbn5ffmfdlrlf7kgearb65srnlasvdjw2jjp3vlwpjl7cjri6sx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2240
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (439040*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (439040*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (439040*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfafhre5ox7tzsvsomayrb2ak5z6xzimey2bw7am77vpvq3u5eh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3512320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2240
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cbayek5j6cbjybpzdb4zrk54q7kmi6yczlwbpv5bkx6q7ofygbuq.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (225792 + x3 + (476672*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (225792 + x3 + (464128*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (225792 + x3 + (451584*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (225792 + x3 + (439040*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x3 + (476672*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (464128*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (451584*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (439040*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qs/cqslsesy3ax3oofw5q4pbjsiw5dareiaqmndefeqle5pksyurejy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2176
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (426496*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (426496*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (426496*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54mc3h4gyxvwe2seskwpowfi3yd4nexu4zrhqmvgrify52m6xz3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3411968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2176
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7qw3ilgqhbax6cfyull2dlhssumfgqkjaqd5jzh362tz6yunsn.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (476672*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (464128*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (451584*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (439040*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (426496*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfs2ehoubzqbi72wniijlprcdj7rmpntt4mdgeardt4ntqe6rde5.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 225792
    x1 = (xindex // 225792)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (200704 + x0 + (476672*x1)), None)
    tmp1 = tl.load(in_ptr1 + (200704 + x0 + (464128*x1)), None)
    tmp3 = tl.load(in_ptr2 + (200704 + x0 + (451584*x1)), None)
    tmp5 = tl.load(in_ptr3 + (200704 + x0 + (439040*x1)), None)
    tmp7 = tl.load(in_ptr4 + (200704 + x0 + (426496*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/cts2n2eb5xqlkkbnoxhw3chev7nx265zksscde5hu33olyua4xps.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (12544 + x3 + (225792*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x3 + (200704*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/candcvkase6isoigwjlkwadae6b5zcoydqvow5njw732zfkfdkvh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2112
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (413952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (413952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (413952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxornmpv22v6ln5onunlwsqjnoire75fmnmf65cg5tdz55f6dyul.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3311616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2112
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwnwyfuyhfimp4my44b5mvxgygc4uk65rvnqg5ue6feoskzl26p.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + (225792*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (200704 + x3 + (413952*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr2 + (x3 + (200704*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (413952*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qy/cqy67v2otrgwc4krjpuihhup33kugpongk6uztccdpr3z4ubgc3z.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (401408*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (401408*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (401408*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, None)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckokbxa26mdsxw6zsrjiekl3qod6w2m7tcv45x7lvm57ap7nmlw5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvv25lyxuixaffttm2wbp2snfzicxowtifggbyashtk4ocrivw5a.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-12544) + x3 + (225792*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (188160 + x3 + (413952*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (188160 + x3 + (401408*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr3 + (x3 + (200704*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (413952*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (401408*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmy2fqj5rjuy6tjcvxbk7ii4dqttw2snx2x3egn2eqf45ukqlxtr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1984
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (388864*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (388864*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (388864*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/condullnswvusrlb5sk5lkombwhjmgvx2gtot2bfj76ozcwl47om.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3110912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1984
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvxkbrnjjdedj5hfuxjyiry72hxtlfkpiq2qs2nsfkyssu2tq5q.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-25088) + x3 + (225792*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (175616 + x3 + (413952*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (175616 + x3 + (401408*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (175616 + x3 + (388864*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr4 + (x3 + (200704*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (413952*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (401408*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (388864*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cibsokjvz5ovecw3bdmlp6ccgz5a7f62rygfcg77744ujyzw7g4j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (376320*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (376320*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (376320*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3342r3sxgly4lm2ai4ajrsrdz6nakntifwzxp3ypfjua3uxhq4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1920
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/con4rld5vz2abdgr5pbaew35kol6qrxlr2zyqnhvknc3y3r7mmgg.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (413952*x1)), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (401408*x1)), None)
    tmp5 = tl.load(in_ptr2 + (x0 + (388864*x1)), None)
    tmp7 = tl.load(in_ptr3 + (x0 + (376320*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccuoywg2jw224i65kik6dlxil2e3h4ng7qtr4jhcy32liyprhiic.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 175616
    x1 = (xindex // 175616)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (225792*x1)), None)
    tmp1 = tl.load(in_ptr1 + (200704 + x0 + (413952*x1)), None)
    tmp3 = tl.load(in_ptr2 + (200704 + x0 + (401408*x1)), None)
    tmp5 = tl.load(in_ptr3 + (200704 + x0 + (388864*x1)), None)
    tmp7 = tl.load(in_ptr4 + (200704 + x0 + (376320*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j4/cj4ogtzfhjpo2fhkh3eo7yviyvcrb3ockxqwyqnieqiwovfjmgpm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-37632) + x3 + (175616*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x3 + (200704*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2ibwtbua7jlc6kmomrcfxb4kmoihji6htgfc6yh4mr6nmctbwu5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1856
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (363776*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (363776*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (363776*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zh4ypcooyf2diqn65piywpkbmnwn4g52lwfzkucjp3tmuqqpmd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2910208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1856
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgifp6fvaotfyrvrof2jenmy7u6ydfnnvysfhcgqne7g5utdek7.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-50176) + x3 + (175616*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (150528 + x3 + (363776*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr2 + (x3 + (200704*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (363776*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahb7kjxdzbqvol3pkd4lbnyllftrggtuugfr74zn7dfdsggotgh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1792
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3jfbgatb7njfzfvi3lskntxjz6thvbxj7gnypf7rvwoq5kuvaa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1792
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dckb2u4pgi5e2lmooaorkqzfy6kd76glcx7lvkevsbm2dqutym.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-62720) + x3 + (175616*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (137984 + x3 + (363776*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (137984 + x3 + (351232*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr3 + (x3 + (200704*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (363776*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (351232*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpviijdtrr7huuufglofd2zbjp3vmkfjy36h5nqobyjey75t26ht.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1728
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2jagxy3rbmh7zwtt7glppqxa2g7js7gtsp5sbo5j4ttjoesg7ma.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1728
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qd3bmo5uq7s7ombeghue7cnckznk5epjataitegndrceiwseeq.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-75264) + x3 + (175616*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (125440 + x3 + (363776*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (125440 + x3 + (351232*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (125440 + x3 + (338688*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr4 + (x3 + (200704*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (363776*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (351232*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (338688*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqnyt3ph5szob7u3sijhnbz5xws6fjlhnyct74xflqlh6eihyeu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (326144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (326144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (326144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphahzfxi57gn5tizhy755jq5l3wqoxoxs7d2ls6jbjoxj54wsbf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1664
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2e3wrxhjmlhpdo7p364dh2b2mumunkf5up5cgyx5p52pomj3sn.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (363776*x1)), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (351232*x1)), None)
    tmp5 = tl.load(in_ptr2 + (x0 + (338688*x1)), None)
    tmp7 = tl.load(in_ptr3 + (x0 + (326144*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjxhbuprigmv3o64z3vtepmqjlnfqcyte5wwkufhzf5h7suhzwt.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 125440
    x1 = (xindex // 125440)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (175616*x1)), None)
    tmp1 = tl.load(in_ptr1 + (200704 + x0 + (363776*x1)), None)
    tmp3 = tl.load(in_ptr2 + (200704 + x0 + (351232*x1)), None)
    tmp5 = tl.load(in_ptr3 + (200704 + x0 + (338688*x1)), None)
    tmp7 = tl.load(in_ptr4 + (200704 + x0 + (326144*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nbncliotgngvbuj5veqxmawy2wprmasjdz3tphqkzpcjt3k7zm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-87808) + x3 + (125440*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x3 + (200704*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cfl2frfz7ovdwqaspj5dfgjg4u6fi35qqj2og5elaxim5k7wwvhq.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-100352) + x3 + (125440*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (100352 + x3 + (313600*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr2 + (x3 + (200704*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (313600*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijdi6pizrazn33pfxpsleu3h2opopxjohkwmlp4r4wg6acdv3em.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygydctr2t3jw6ed7usce6ls6gec3l3exvbvfajd7em65fwl2lta.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5o/c5ozmk52tijepjjt6njieun5rqoqoifz54u3zjmwq5uekwwtai3l.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-112896) + x3 + (125440*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (87808 + x3 + (313600*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (87808 + x3 + (301056*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr3 + (x3 + (200704*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (313600*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (301056*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgmx4ae7cuhaqevx2teph452arnwaxn5366mx5yymi6yvusci5z.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1472
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (288512*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (288512*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (288512*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/creruvk5bniyvis43kovehnedxi44rdzvnq3vkqux4uzwr5ccy4h.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2308096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1472
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3yzkx4iypm4jhwzlbhh6buy4x6adsvr4rp4dzvwi4tkbixx33v.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-125440) + x3 + (125440*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (75264 + x3 + (313600*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (75264 + x3 + (301056*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (75264 + x3 + (288512*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr4 + (x3 + (200704*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (313600*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (301056*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (288512*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2eot444s6cathfaqdq22yalzformpkf7asqnh3qvuz25cdxho4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1408
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (275968*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (275968*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (275968*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvoxxrenmpfqw2xyfswln4uyycvxy23trbpjo72kupehxwlmj5fr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2207744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1408
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmrbwho344rkmoic7eitd7ig7upso7dncdnj7whrlgeapefdm35.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (313600*x1)), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (301056*x1)), None)
    tmp5 = tl.load(in_ptr2 + (x0 + (288512*x1)), None)
    tmp7 = tl.load(in_ptr3 + (x0 + (275968*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckof5vuv6ov7tknnstqelratrvgvcqkdmhtaqre73htya7agvvue.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 75264
    x1 = (xindex // 75264)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (125440*x1)), None)
    tmp1 = tl.load(in_ptr1 + (200704 + x0 + (313600*x1)), None)
    tmp3 = tl.load(in_ptr2 + (200704 + x0 + (301056*x1)), None)
    tmp5 = tl.load(in_ptr3 + (200704 + x0 + (288512*x1)), None)
    tmp7 = tl.load(in_ptr4 + (200704 + x0 + (275968*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citivmzbbkdxizxr3lkz4bdyjlx3b6yomd27bh6t4opgk5d4k5hm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-137984) + x3 + (75264*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x3 + (200704*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/ccz7i676x7q6lmczyvirgpsc5ndoyytuad5lfumhmbuny2wbvwtb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (263424*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (263424*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (263424*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zu5zrt6e5dvdg65j7pqkiorz4vbbm27jwvvczoj6tfeaqc3z5d.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1344
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clf73bqlxkpzbbw56fzfst7mvihwfpaksulhlkgppozycd4fbb4z.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-150528) + x3 + (75264*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (50176 + x3 + (263424*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr2 + (x3 + (200704*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (263424*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdymftsdngcwlsb4kpzxglfy3ygxsub6gfqns6cqu3757onne6so.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (250880*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (250880*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (250880*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4lqd3dglfmrilaaeehd36jpobbettc3bs3ajoav5qw2ximlqop.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1280
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccy6hftkuvsmjg5gptlxlxvlkisaezppthaftwtocsmi3q7h6hgm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-163072) + x3 + (75264*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (37632 + x3 + (263424*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (37632 + x3 + (250880*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr3 + (x3 + (200704*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (263424*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (250880*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2m/c2m3jvertznz7zeny642tb33akjtcxmwt32jjkxpbt4eqnn6dpp6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1216
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (238336*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (238336*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (238336*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbcvqurjv7mwxjp3xkn4urdegawscehiecih5r3fnush2r4vtkz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1906688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1216
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czkuyobg57du5fwo54rts36oic5g7u7s47fodfwtzaxf5ioanu5v.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-175616) + x3 + (75264*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (25088 + x3 + (263424*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (25088 + x3 + (250880*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (25088 + x3 + (238336*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr4 + (x3 + (200704*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (263424*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (250880*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (238336*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceynlur44gtotpd32kvyuqnwhqxyrs6iwsm6ypaf46bgdg63rnzu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 800
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (627200*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (627200*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (627200*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csox6p3wmbmca7bhd5gbenqt7a7uzmfwrqt26xrktvzk3ezba75b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 800
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wr3ph4xizicp2ifaaavhekfs7tfjmpjexwtdc3dznu264dv7iz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1152
    x2 = (xindex // 225792)
    x3 = xindex % 225792
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-200704) + x3 + (75264*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x3 + (263424*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x3 + (250880*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (x3 + (238336*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr4 + (x3 + (200704*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (263424*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (250880*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (238336*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtxid2h7mcwvvoy7pluqlkccs47gp3czpalwgshrvbflaura37x.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp16 = tmp15 <= tmp1
        tmp18 = tl.where(tmp16, tmp1, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp22 = tmp18 * tmp10
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp13 * tmp26
    tmp28 = tmp24 * tmp26
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr5 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndedrsbbisv2wy55o7liqajus3z2thqyza34yehk2gdayknhxxy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7225344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1152
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x3), None)
    tmp24 = tl.load(in_ptr8 + (x3), None)
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22 <= tmp1
    tmp25 = tl.where(tmp23, tmp1, tmp24)
    tmp27 = tmp26 * tmp9
    tmp28 = tmp27 * tmp12
    tmp29 = tmp7 * tmp28
    tmp30 = tmp25 - tmp29
    tmp32 = tmp31 * tmp9
    tmp33 = tmp30 - tmp32
    tmp35 = tmp11 * tmp34
    tmp36 = tmp33 * tmp35
    tmp37 = tmp21 + tmp36
    tl.store(in_out_ptr0 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6eksj4aogyo57ktxqjsmkytmrocnbkfxtmm2cadx5bldo4hvsnm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (451584 + x3 + (903168*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x3 + (903168*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hz/chzrnt6sgkm7kblxpraf4vuco3pjcvg4pqzzopkondvriacaa7tv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 400
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yo/cyofudfe7ychsppgznfetoegidsnstiqfudbuewngwji5wkglwuf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 400
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7l6dwcqmz3avz25ucttz2vetyrb6dezobdkfl5q3mecjx643pms.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1088
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (852992*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (852992*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (852992*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7xvjlxqe5t5kzkszbjxgkgntici3tqg2mkyt2k6gt7sug4s57o.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6823936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1088
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cerkozjmaixwlylulxhjn6p56dzpk6psoaloytt56jv3psf76gz2.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (401408 + x3 + (903168*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (401408 + x3 + (852992*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x3 + (903168*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (852992*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vzfmoekd5ehjv2s3soo4eknqh2c54wjajiqktzydu54xlcvx3a.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/chajskpyxebmuue3dpj5qplrqhjjd7m544q73t3dlwfktjc3b6kt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_95', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvw46hz2k56l462qr522kwovbknx6ymgjutkkntkjvthxao6o3fh.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (351232 + x3 + (903168*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (351232 + x3 + (852992*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (351232 + x3 + (802816*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x3 + (903168*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (852992*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (802816*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y3/cy34vwnyw5iv6kxs2pnsjnaiytbcpnh74jril3iyyf5ab2x6biuf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (752640*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (752640*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (752640*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdw5nswnjxusfkq624t6ojcqd6ua3nwh7enxqsumdhh6kadsm5l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 960
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbntmtw62lut4jrwiiium7nwxbijgo4or523eqnhubypf3yfavh.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (301056 + x3 + (903168*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (301056 + x3 + (852992*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (301056 + x3 + (802816*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (301056 + x3 + (752640*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x3 + (903168*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (852992*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (802816*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (752640*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uk/cuk3pz6wma7ymbbinvflsgp2u6zbjvvoz2picljlawqgwzliunbk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (702464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (702464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (702464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwexfjx3y2wuntlhclmnq2bzurcb3ijyvyhhoks7csu2pgc3lu3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2lciicrag2eej6y7x2zt2zgxrwqwtd4v263mx4xqwodayxvdv56.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 401408
    x1 = (xindex // 401408)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (903168*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (852992*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (802816*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (752640*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (702464*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/cae7gielro6jixji3ibeggptcnu6nsaq5busxog6e6bfe6u4c7w4.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_103 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 301056
    x1 = (xindex // 301056)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (401408 + x0 + (903168*x1)), None)
    tmp1 = tl.load(in_ptr1 + (401408 + x0 + (852992*x1)), None)
    tmp3 = tl.load(in_ptr2 + (401408 + x0 + (802816*x1)), None)
    tmp5 = tl.load(in_ptr3 + (401408 + x0 + (752640*x1)), None)
    tmp7 = tl.load(in_ptr4 + (401408 + x0 + (702464*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpe5u75ufrohsue53lvjuqquic3kztgxo7mt7mwtdjepkhkad7gw.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-150528) + x3 + (301056*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x3 + (401408*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54dleattng5sklpcfo57w3qollvsabasjizhkxp77bzqet6ar3v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (652288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (652288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (652288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpx5hmaxtfs3kjpt3rialp4h7dbgjg6codi3qpg24iiln4jejv7d.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_106', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5218304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 832
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgv5veqwjvq2jfsioleyt3ngxl2dzna52lfludm42vxyht5odqkn.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_107 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-200704) + x3 + (301056*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (200704 + x3 + (652288*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr2 + (x3 + (401408*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (652288*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdfpkmlj3w7iz27fldu4dx43ibxo4xwstttmqeb6cad2s2yopsv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_108 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (602112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (602112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (602112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrusndkriccbxvad2it2j2ob2ywrgyvbxwbcgnjcchowath6b6i.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_109 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_109', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdbi7hoqcmcevbhw7lafqrtcrvzmie5t466nkbdfrd5fjzuwg2d.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-250880) + x3 + (301056*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (150528 + x3 + (652288*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (150528 + x3 + (602112*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr3 + (x3 + (401408*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (652288*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (602112*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7wubkxuqn32o3kuawpww6hh26d7gzzngo747qpm7f3vfikgwqs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 704
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (551936*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (551936*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (551936*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsgpqnnh36y4tboedbophkntmryfifngb62xbidzmzsvb274fow.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_112', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4415488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 704
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crghlkxc7axks4k7hhomcynpk7cwdoyipvtyjyqbvtylwb5qwn7y.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_113 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 576
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-301056) + x3 + (301056*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (100352 + x3 + (652288*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (100352 + x3 + (602112*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (100352 + x3 + (551936*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr4 + (x3 + (401408*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (652288*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (602112*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (551936*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3fpwig5ej5zc3vhderhvvyyiptuwddlycwd54wiu6bguu73wlt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_114 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 400
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1254400*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (1254400*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (1254400*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxbgaasrelpmu3pyxqezaoswen5ubuday34v7bkzs3rwemxkftc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_115', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10035200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 400
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygdmslilc2pa3a5xlolv4trnr52ubhld6drsixfze5kxd2hjkrz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_116 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_116', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 640
    x2 = (xindex // 501760)
    x3 = xindex % 501760
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 512, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-401408) + x3 + (301056*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x3 + (652288*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x3 + (602112*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (x3 + (551936*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr4 + (x3 + (401408*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (652288*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (602112*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (551936*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp55bpoarrhfunaxy7qbrxq7abksba7o5f3dx6xpyf635jqxtvia.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_117 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 376
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1179136*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (1179136*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (1179136*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r1 + (3136*x0) + (1179136*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (3136*x0) + (1179136*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp16 = tmp15 <= tmp1
        tmp18 = tl.where(tmp16, tmp1, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp22 = tmp18 * tmp10
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp24, xmask)
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp13 * tmp26
    tmp28 = tmp24 * tmp26
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr5 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ri/cri6ujvqe2yjsn3amkz32gwvm66qgbvazvwra3rwlpzaqtltbgea.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_118 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_118', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9433088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 376
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x3), None)
    tmp24 = tl.load(in_ptr8 + (x3), None)
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22 <= tmp1
    tmp25 = tl.where(tmp23, tmp1, tmp24)
    tmp27 = tmp26 * tmp9
    tmp28 = tmp27 * tmp12
    tmp29 = tmp7 * tmp28
    tmp30 = tmp25 - tmp29
    tmp32 = tmp31 * tmp9
    tmp33 = tmp30 - tmp32
    tmp35 = tmp11 * tmp34
    tmp36 = tmp33 * tmp35
    tmp37 = tmp21 + tmp36
    tl.store(in_out_ptr0 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dmig3e5szplxpw3z5i2o22yxqm6ag7muc7ryhxasmnthxdynzd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_119 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_119', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6924288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 276
    x2 = (xindex // 865536)
    x3 = xindex % 865536
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (313600 + x3 + (1179136*x2)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x3 + (1179136*x2)), tmp8, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cjeucua6bhibbnwo7av56ktmgew2rmkgkvb2womlk4itxb46drtu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_120 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 800
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 200
    x1 = (xindex // 200)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (627200*(r2 // 3136)) + (1254400*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (627200*(r2 // 3136)) + (1254400*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (627200*(r2 // 3136)) + (1254400*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckegkhdvpmrz7aghv644v4rv7qtngln4kwvra2gqym52bcsl2uzv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_121 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnd2qlgxek27l6v2xmnatzdx2mc7p23owiytecx5mjnzcajlv2r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_122 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_122', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqw3b5ejlfzzantqhd6tvvzrnt4sziodcaj3eyf73j6vgzjgxedf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 200
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2lmygoeh55f7a7rcpejpgybjpiri3kvyh5uig5xtw5gfqkrjda.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_124 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_124', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 356
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1116416*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (1116416*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (1116416*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkxcpqvkxjtarpjrd2nhlalxq5mxt7o5645xr23yhaj36327sbq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_125 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_125', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8931328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 356
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l3/cl3sskxirjhvlyukicfx3gtaicqnihhqw5o2ju3d7bljjntp6wwd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_126 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_126', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6924288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 276
    x2 = (xindex // 865536)
    x3 = xindex % 865536
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (250880 + x3 + (1179136*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (250880 + x3 + (1116416*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x3 + (1179136*x2)), tmp10, other=0.0)
    tmp12 = tl.load(in_ptr1 + (x3 + (1116416*x2)), tmp10, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zz/czzpvnbl6jgnfkvhcwck6oz47y2fnhew57riz7ja3jrzktdek5bp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_127 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1053696*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (1053696*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (1053696*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmqai3pe74e3s6v7u3ho2zzamcjh7ind3bti5wvn3jtfxsh2d7l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_128 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_128', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8429568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 336
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxnbybns6zi5oeh5bxz3hcxygor2sknmwqawu6xuseudxbpkp7o.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_129 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6924288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 276
    x2 = (xindex // 865536)
    x3 = xindex % 865536
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (188160 + x3 + (1179136*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (188160 + x3 + (1116416*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (188160 + x3 + (1053696*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x3 + (1179136*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (1116416*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (1053696*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clris4en3if2pukqr5nzszdbens2yi36fevc2j6lg3dgvuf6vvzc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_130 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 316
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (990976*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (990976*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (990976*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxdt55n7pep6637ni5pzj53oat7mnpowysw4gf6xea6k7xvui2p.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_131', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7927808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 316
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pw/cpwomawapjf3uzdqhibux3k2b5wd4a55z4n5lqm3zrsxrqfr7gnz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_132 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_132', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6924288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 276
    x2 = (xindex // 865536)
    x3 = xindex % 865536
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (125440 + x3 + (1179136*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (125440 + x3 + (1116416*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (125440 + x3 + (1053696*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (125440 + x3 + (990976*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x3 + (1179136*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (1116416*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (1053696*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (990976*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/cao6aujppmatgzusc4hfkltnocgwan2ppsh44gksubbd7h3rpkds.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_slice_backward_133 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_133', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7426048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 296
    x2 = (xindex // 928256)
    x3 = xindex % 928256
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + (1179136*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x3 + (1116416*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x3 + (1053696*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (x3 + (990976*x2)), tmp2, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x3 + (1179136*x2)), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr1 + (x3 + (1116416*x2)), tmp14, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x3 + (1053696*x2)), tmp14, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (990976*x2)), tmp14, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tl.store(out_ptr0 + (x4), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfeq6nfs4epjiykbdmgjro724palimi65otnqlpye6cver2ryewe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_134 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp14 = tmp13 <= tmp1
        tmp16 = tl.where(tmp14, tmp1, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp20 = tmp16 * tmp8
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp18, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbug5jpml22eksxzzefgmz3v36rfn42ecmvqie6gqeoekvmmlql6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_135 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwybcgyyrcyipi6d6bzpebt46qg4s5seo5mlbamyszratkqwa7lm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_136 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_136', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp10
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tl.store(out_ptr3 + (x0), tmp12, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c67ozgwdgmu66l5kyzdvcyvryiuknl3wab7sy54huvm5nqb7prwe.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_137 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_137', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x3), None)
    tmp22 = tl.load(in_ptr7 + (x3), None)
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = 3.985969387755102e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp4 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20 <= tmp1
    tmp23 = tl.where(tmp21, tmp1, tmp22)
    tmp25 = tmp24 * tmp7
    tmp26 = tmp25 * tmp10
    tmp27 = tmp5 * tmp26
    tmp28 = tmp23 - tmp27
    tmp30 = tmp29 * tmp7
    tmp31 = tmp28 - tmp30
    tmp33 = tmp9 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp19 + tmp34
    tl.store(in_out_ptr0 + (x3), tmp35, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqobqvx2mz5jfkkatmcq6tk7sxqhwdirjessnodfz3ztfnd5u2m.py
# Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]

triton_poi_fused_max_pool2d_with_indices_backward_138 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_backward_138', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112) % 112
    x2 = (xindex // 12544)
    x3 = xindex % 12544
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp7 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp19 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp31 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (x1 // 2))
    tmp10 = tl.math.min(56, 1 + ((1 + x1) // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp13 = tl.math.min(56, 1 + ((1 + x0) // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (x0 // 2))
    tmp25 = tmp24 < tmp13
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp21
    tmp28 = tmp18 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp18)
    tmp32 = tmp30 == tmp2
    tmp33 = tmp23 & tmp14
    tmp34 = tmp33 & tmp32
    tmp35 = tmp29 + tmp31
    tmp36 = tl.where(tmp34, tmp35, tmp29)
    tl.store(out_ptr0 + (x5), tmp36, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtam64ou2zkktxtviar2663caqfvwuglsob62avj5foun5krvyk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_139 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_139', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x0) + (1605632*(r2 // 12544)) + (3211264*x1) + (r2 % 12544)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (1605632*(r2 // 12544)) + (3211264*x1) + (r2 % 12544)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((12544*x0) + (1605632*(r2 // 12544)) + (3211264*x1) + (r2 % 12544)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rzeaunyetsfemv72m44ktbcfp62tg72ecrflykur3cnz7ytuot.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_140 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_140', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdr6cevfgwdtzmcujzauz7uc6sn3pvdyfrsuyphm2yfsckbyqqpj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_141 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_141', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 9.964923469387754e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_668, convolution, squeeze_1, relu, getitem_3, squeeze_4, relu_1, relu_2, convolution_2, squeeze_10, relu_3, convolution_3, squeeze_13, relu_4, cat_1, squeeze_16, relu_5, convolution_5, squeeze_19, relu_6, convolution_6, squeeze_22, relu_7, cat_3, squeeze_25, relu_8, convolution_8, squeeze_28, relu_9, convolution_9, squeeze_31, relu_10, cat_5, squeeze_34, relu_11, convolution_11, squeeze_37, relu_12, convolution_12, squeeze_40, relu_13, cat_7, squeeze_43, relu_14, relu_15, convolution_15, squeeze_49, relu_16, convolution_16, squeeze_52, relu_17, cat_9, squeeze_55, relu_18, convolution_18, squeeze_58, relu_19, convolution_19, squeeze_61, relu_20, cat_11, squeeze_64, relu_21, convolution_21, squeeze_67, relu_22, convolution_22, squeeze_70, relu_23, cat_13, squeeze_73, relu_24, convolution_24, squeeze_76, relu_25, convolution_25, squeeze_79, relu_26, cat_15, squeeze_82, relu_27, convolution_27, squeeze_85, relu_28, convolution_28, squeeze_88, relu_29, cat_17, squeeze_91, relu_30, convolution_30, squeeze_94, relu_31, convolution_31, squeeze_97, relu_32, cat_19, squeeze_100, relu_33, convolution_33, squeeze_103, relu_34, convolution_34, squeeze_106, relu_35, cat_21, squeeze_109, relu_36, convolution_36, squeeze_112, relu_37, convolution_37, squeeze_115, relu_38, cat_23, squeeze_118, relu_39, relu_40, convolution_40, squeeze_124, relu_41, convolution_41, squeeze_127, relu_42, cat_25, squeeze_130, relu_43, convolution_43, squeeze_133, relu_44, convolution_44, squeeze_136, relu_45, cat_27, squeeze_139, relu_46, convolution_46, squeeze_142, relu_47, convolution_47, squeeze_145, relu_48, cat_29, squeeze_148, relu_49, convolution_49, squeeze_151, relu_50, convolution_50, squeeze_154, relu_51, cat_31, squeeze_157, relu_52, convolution_52, squeeze_160, relu_53, convolution_53, squeeze_163, relu_54, cat_33, squeeze_166, relu_55, convolution_55, squeeze_169, relu_56, convolution_56, squeeze_172, relu_57, cat_35, squeeze_175, relu_58, convolution_58, squeeze_178, relu_59, convolution_59, squeeze_181, relu_60, cat_37, squeeze_184, relu_61, convolution_61, squeeze_187, relu_62, convolution_62, squeeze_190, relu_63, cat_39, squeeze_193, relu_64, convolution_64, squeeze_196, relu_65, convolution_65, squeeze_199, relu_66, cat_41, squeeze_202, relu_67, convolution_67, squeeze_205, relu_68, convolution_68, squeeze_208, relu_69, cat_43, squeeze_211, relu_70, convolution_70, squeeze_214, relu_71, convolution_71, squeeze_217, relu_72, cat_45, squeeze_220, relu_73, convolution_73, squeeze_223, relu_74, convolution_74, squeeze_226, relu_75, cat_47, squeeze_229, relu_76, convolution_76, squeeze_232, relu_77, convolution_77, squeeze_235, relu_78, cat_49, squeeze_238, relu_79, convolution_79, squeeze_241, relu_80, convolution_80, squeeze_244, relu_81, cat_51, squeeze_247, relu_82, convolution_82, squeeze_250, relu_83, convolution_83, squeeze_253, relu_84, cat_53, squeeze_256, relu_85, convolution_85, squeeze_259, relu_86, convolution_86, squeeze_262, relu_87, cat_55, squeeze_265, relu_88, convolution_88, squeeze_268, relu_89, convolution_89, squeeze_271, relu_90, cat_57, squeeze_274, relu_91, convolution_91, squeeze_277, relu_92, convolution_92, squeeze_280, relu_93, cat_59, squeeze_283, relu_94, convolution_94, squeeze_286, relu_95, convolution_95, squeeze_289, relu_96, cat_61, squeeze_292, relu_97, convolution_97, squeeze_295, relu_98, convolution_98, squeeze_298, relu_99, cat_63, squeeze_301, relu_100, relu_101, convolution_101, squeeze_307, relu_102, convolution_102, squeeze_310, relu_103, cat_65, squeeze_313, relu_104, convolution_104, squeeze_316, relu_105, convolution_105, squeeze_319, relu_106, cat_67, squeeze_322, relu_107, convolution_107, squeeze_325, relu_108, convolution_108, squeeze_328, relu_109, cat_69, squeeze_331, mean, le, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1682, unsqueeze_1694, unsqueeze_1706, unsqueeze_1718, unsqueeze_1730, sub_543, unsqueeze_1766, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (200, ), (1, ))
    assert_size_stride(primals_9, (200, ), (1, ))
    assert_size_stride(primals_11, (316, ), (1, ))
    assert_size_stride(primals_13, (200, ), (1, ))
    assert_size_stride(primals_15, (200, ), (1, ))
    assert_size_stride(primals_17, (336, ), (1, ))
    assert_size_stride(primals_19, (200, ), (1, ))
    assert_size_stride(primals_21, (200, ), (1, ))
    assert_size_stride(primals_23, (356, ), (1, ))
    assert_size_stride(primals_25, (200, ), (1, ))
    assert_size_stride(primals_27, (200, ), (1, ))
    assert_size_stride(primals_29, (376, ), (1, ))
    assert_size_stride(primals_31, (376, ), (1, ))
    assert_size_stride(primals_33, (400, ), (1, ))
    assert_size_stride(primals_35, (400, ), (1, ))
    assert_size_stride(primals_37, (704, ), (1, ))
    assert_size_stride(primals_39, (400, ), (1, ))
    assert_size_stride(primals_41, (400, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (400, ), (1, ))
    assert_size_stride(primals_47, (400, ), (1, ))
    assert_size_stride(primals_49, (832, ), (1, ))
    assert_size_stride(primals_51, (400, ), (1, ))
    assert_size_stride(primals_53, (400, ), (1, ))
    assert_size_stride(primals_55, (896, ), (1, ))
    assert_size_stride(primals_57, (400, ), (1, ))
    assert_size_stride(primals_59, (400, ), (1, ))
    assert_size_stride(primals_61, (960, ), (1, ))
    assert_size_stride(primals_63, (400, ), (1, ))
    assert_size_stride(primals_65, (400, ), (1, ))
    assert_size_stride(primals_67, (1024, ), (1, ))
    assert_size_stride(primals_69, (400, ), (1, ))
    assert_size_stride(primals_71, (400, ), (1, ))
    assert_size_stride(primals_73, (1088, ), (1, ))
    assert_size_stride(primals_75, (400, ), (1, ))
    assert_size_stride(primals_77, (400, ), (1, ))
    assert_size_stride(primals_79, (1152, ), (1, ))
    assert_size_stride(primals_81, (1152, ), (1, ))
    assert_size_stride(primals_83, (800, ), (1, ))
    assert_size_stride(primals_85, (800, ), (1, ))
    assert_size_stride(primals_87, (1216, ), (1, ))
    assert_size_stride(primals_89, (800, ), (1, ))
    assert_size_stride(primals_91, (800, ), (1, ))
    assert_size_stride(primals_93, (1280, ), (1, ))
    assert_size_stride(primals_95, (800, ), (1, ))
    assert_size_stride(primals_97, (800, ), (1, ))
    assert_size_stride(primals_99, (1344, ), (1, ))
    assert_size_stride(primals_101, (800, ), (1, ))
    assert_size_stride(primals_103, (800, ), (1, ))
    assert_size_stride(primals_105, (1408, ), (1, ))
    assert_size_stride(primals_107, (800, ), (1, ))
    assert_size_stride(primals_109, (800, ), (1, ))
    assert_size_stride(primals_111, (1472, ), (1, ))
    assert_size_stride(primals_113, (800, ), (1, ))
    assert_size_stride(primals_115, (800, ), (1, ))
    assert_size_stride(primals_117, (1536, ), (1, ))
    assert_size_stride(primals_119, (800, ), (1, ))
    assert_size_stride(primals_121, (800, ), (1, ))
    assert_size_stride(primals_123, (1600, ), (1, ))
    assert_size_stride(primals_125, (800, ), (1, ))
    assert_size_stride(primals_127, (800, ), (1, ))
    assert_size_stride(primals_129, (1664, ), (1, ))
    assert_size_stride(primals_131, (800, ), (1, ))
    assert_size_stride(primals_133, (800, ), (1, ))
    assert_size_stride(primals_135, (1728, ), (1, ))
    assert_size_stride(primals_137, (800, ), (1, ))
    assert_size_stride(primals_139, (800, ), (1, ))
    assert_size_stride(primals_141, (1792, ), (1, ))
    assert_size_stride(primals_143, (800, ), (1, ))
    assert_size_stride(primals_145, (800, ), (1, ))
    assert_size_stride(primals_147, (1856, ), (1, ))
    assert_size_stride(primals_149, (800, ), (1, ))
    assert_size_stride(primals_151, (800, ), (1, ))
    assert_size_stride(primals_153, (1920, ), (1, ))
    assert_size_stride(primals_155, (800, ), (1, ))
    assert_size_stride(primals_157, (800, ), (1, ))
    assert_size_stride(primals_159, (1984, ), (1, ))
    assert_size_stride(primals_161, (800, ), (1, ))
    assert_size_stride(primals_163, (800, ), (1, ))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_167, (800, ), (1, ))
    assert_size_stride(primals_169, (800, ), (1, ))
    assert_size_stride(primals_171, (2112, ), (1, ))
    assert_size_stride(primals_173, (800, ), (1, ))
    assert_size_stride(primals_175, (800, ), (1, ))
    assert_size_stride(primals_177, (2176, ), (1, ))
    assert_size_stride(primals_179, (800, ), (1, ))
    assert_size_stride(primals_181, (800, ), (1, ))
    assert_size_stride(primals_183, (2240, ), (1, ))
    assert_size_stride(primals_185, (800, ), (1, ))
    assert_size_stride(primals_187, (800, ), (1, ))
    assert_size_stride(primals_189, (2304, ), (1, ))
    assert_size_stride(primals_191, (800, ), (1, ))
    assert_size_stride(primals_193, (800, ), (1, ))
    assert_size_stride(primals_195, (2368, ), (1, ))
    assert_size_stride(primals_197, (800, ), (1, ))
    assert_size_stride(primals_199, (800, ), (1, ))
    assert_size_stride(primals_201, (2432, ), (1, ))
    assert_size_stride(primals_203, (2432, ), (1, ))
    assert_size_stride(primals_205, (1600, ), (1, ))
    assert_size_stride(primals_207, (1600, ), (1, ))
    assert_size_stride(primals_209, (2432, ), (1, ))
    assert_size_stride(primals_211, (1600, ), (1, ))
    assert_size_stride(primals_213, (1600, ), (1, ))
    assert_size_stride(primals_215, (2560, ), (1, ))
    assert_size_stride(primals_217, (1600, ), (1, ))
    assert_size_stride(primals_219, (1600, ), (1, ))
    assert_size_stride(primals_221, (2688, ), (1, ))
    assert_size_stride(primals_223, (128, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_224, (296, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_225, (200, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_226, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_227, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_228, (200, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(primals_229, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_230, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_231, (200, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_232, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_233, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_234, (200, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(primals_235, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_236, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_237, (640, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(primals_238, (400, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(primals_239, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_240, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_241, (400, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_242, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_243, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_244, (400, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_245, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_246, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_247, (400, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_248, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_249, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_250, (400, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_251, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_252, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_253, (400, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_254, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_255, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_256, (400, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_257, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_258, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_259, (400, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_260, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_261, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_262, (1152, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_263, (800, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_264, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_265, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_266, (800, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(primals_267, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_268, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_269, (800, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_270, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_271, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_272, (800, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_273, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_274, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_275, (800, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(primals_276, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_277, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_278, (800, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(primals_279, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_280, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_281, (800, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_282, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_283, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_284, (800, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_285, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_286, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_287, (800, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(primals_288, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_289, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_290, (800, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(primals_291, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_292, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_293, (800, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(primals_294, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_295, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_296, (800, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(primals_297, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_298, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_299, (800, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_300, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_301, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_302, (800, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(primals_303, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_304, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_305, (800, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_306, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_307, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_308, (800, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(primals_309, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_310, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_311, (800, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(primals_312, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_313, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_314, (800, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_315, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_316, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_317, (800, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_318, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_319, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_320, (800, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(primals_321, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_322, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_323, (2304, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_324, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_325, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_326, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_327, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_328, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_329, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_330, (1600, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(primals_331, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_332, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_333, (1000, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(primals_668, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 128, 112, 112), (1605632, 12544, 112, 1))
    assert_size_stride(squeeze_1, (128, ), (1, ))
    assert_size_stride(relu, (8, 128, 112, 112), (1605632, 12544, 112, 1))
    assert_size_stride(getitem_3, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(squeeze_4, (128, ), (1, ))
    assert_size_stride(relu_1, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_2, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_2, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_10, (200, ), (1, ))
    assert_size_stride(relu_3, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(convolution_3, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_13, (200, ), (1, ))
    assert_size_stride(relu_4, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(cat_1, (8, 316, 56, 56), (990976, 3136, 56, 1))
    assert_size_stride(squeeze_16, (316, ), (1, ))
    assert_size_stride(relu_5, (8, 316, 56, 56), (990976, 3136, 56, 1))
    assert_size_stride(convolution_5, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_19, (200, ), (1, ))
    assert_size_stride(relu_6, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(convolution_6, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_22, (200, ), (1, ))
    assert_size_stride(relu_7, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(cat_3, (8, 336, 56, 56), (1053696, 3136, 56, 1))
    assert_size_stride(squeeze_25, (336, ), (1, ))
    assert_size_stride(relu_8, (8, 336, 56, 56), (1053696, 3136, 56, 1))
    assert_size_stride(convolution_8, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_28, (200, ), (1, ))
    assert_size_stride(relu_9, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(convolution_9, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_31, (200, ), (1, ))
    assert_size_stride(relu_10, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(cat_5, (8, 356, 56, 56), (1116416, 3136, 56, 1))
    assert_size_stride(squeeze_34, (356, ), (1, ))
    assert_size_stride(relu_11, (8, 356, 56, 56), (1116416, 3136, 56, 1))
    assert_size_stride(convolution_11, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_37, (200, ), (1, ))
    assert_size_stride(relu_12, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(convolution_12, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(squeeze_40, (200, ), (1, ))
    assert_size_stride(relu_13, (8, 200, 56, 56), (627200, 3136, 56, 1))
    assert_size_stride(cat_7, (8, 376, 56, 56), (1179136, 3136, 56, 1))
    assert_size_stride(squeeze_43, (376, ), (1, ))
    assert_size_stride(relu_14, (8, 376, 56, 56), (1179136, 3136, 56, 1))
    assert_size_stride(relu_15, (8, 376, 56, 56), (1179136, 3136, 56, 1))
    assert_size_stride(convolution_15, (8, 400, 56, 56), (1254400, 3136, 56, 1))
    assert_size_stride(squeeze_49, (400, ), (1, ))
    assert_size_stride(relu_16, (8, 400, 56, 56), (1254400, 3136, 56, 1))
    assert_size_stride(convolution_16, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_52, (400, ), (1, ))
    assert_size_stride(relu_17, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_9, (8, 704, 28, 28), (551936, 784, 28, 1))
    assert_size_stride(squeeze_55, (704, ), (1, ))
    assert_size_stride(relu_18, (8, 704, 28, 28), (551936, 784, 28, 1))
    assert_size_stride(convolution_18, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_58, (400, ), (1, ))
    assert_size_stride(relu_19, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(convolution_19, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_61, (400, ), (1, ))
    assert_size_stride(relu_20, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_11, (8, 768, 28, 28), (602112, 784, 28, 1))
    assert_size_stride(squeeze_64, (768, ), (1, ))
    assert_size_stride(relu_21, (8, 768, 28, 28), (602112, 784, 28, 1))
    assert_size_stride(convolution_21, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_67, (400, ), (1, ))
    assert_size_stride(relu_22, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(convolution_22, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_70, (400, ), (1, ))
    assert_size_stride(relu_23, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_13, (8, 832, 28, 28), (652288, 784, 28, 1))
    assert_size_stride(squeeze_73, (832, ), (1, ))
    assert_size_stride(relu_24, (8, 832, 28, 28), (652288, 784, 28, 1))
    assert_size_stride(convolution_24, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_76, (400, ), (1, ))
    assert_size_stride(relu_25, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(convolution_25, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_79, (400, ), (1, ))
    assert_size_stride(relu_26, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_15, (8, 896, 28, 28), (702464, 784, 28, 1))
    assert_size_stride(squeeze_82, (896, ), (1, ))
    assert_size_stride(relu_27, (8, 896, 28, 28), (702464, 784, 28, 1))
    assert_size_stride(convolution_27, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_85, (400, ), (1, ))
    assert_size_stride(relu_28, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(convolution_28, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_88, (400, ), (1, ))
    assert_size_stride(relu_29, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_17, (8, 960, 28, 28), (752640, 784, 28, 1))
    assert_size_stride(squeeze_91, (960, ), (1, ))
    assert_size_stride(relu_30, (8, 960, 28, 28), (752640, 784, 28, 1))
    assert_size_stride(convolution_30, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_94, (400, ), (1, ))
    assert_size_stride(relu_31, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(convolution_31, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_97, (400, ), (1, ))
    assert_size_stride(relu_32, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_19, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_100, (1024, ), (1, ))
    assert_size_stride(relu_33, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_33, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_103, (400, ), (1, ))
    assert_size_stride(relu_34, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(convolution_34, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_106, (400, ), (1, ))
    assert_size_stride(relu_35, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_21, (8, 1088, 28, 28), (852992, 784, 28, 1))
    assert_size_stride(squeeze_109, (1088, ), (1, ))
    assert_size_stride(relu_36, (8, 1088, 28, 28), (852992, 784, 28, 1))
    assert_size_stride(convolution_36, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_112, (400, ), (1, ))
    assert_size_stride(relu_37, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(convolution_37, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(squeeze_115, (400, ), (1, ))
    assert_size_stride(relu_38, (8, 400, 28, 28), (313600, 784, 28, 1))
    assert_size_stride(cat_23, (8, 1152, 28, 28), (903168, 784, 28, 1))
    assert_size_stride(squeeze_118, (1152, ), (1, ))
    assert_size_stride(relu_39, (8, 1152, 28, 28), (903168, 784, 28, 1))
    assert_size_stride(relu_40, (8, 1152, 28, 28), (903168, 784, 28, 1))
    assert_size_stride(convolution_40, (8, 800, 28, 28), (627200, 784, 28, 1))
    assert_size_stride(squeeze_124, (800, ), (1, ))
    assert_size_stride(relu_41, (8, 800, 28, 28), (627200, 784, 28, 1))
    assert_size_stride(convolution_41, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_127, (800, ), (1, ))
    assert_size_stride(relu_42, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_25, (8, 1216, 14, 14), (238336, 196, 14, 1))
    assert_size_stride(squeeze_130, (1216, ), (1, ))
    assert_size_stride(relu_43, (8, 1216, 14, 14), (238336, 196, 14, 1))
    assert_size_stride(convolution_43, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_133, (800, ), (1, ))
    assert_size_stride(relu_44, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_44, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_136, (800, ), (1, ))
    assert_size_stride(relu_45, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_27, (8, 1280, 14, 14), (250880, 196, 14, 1))
    assert_size_stride(squeeze_139, (1280, ), (1, ))
    assert_size_stride(relu_46, (8, 1280, 14, 14), (250880, 196, 14, 1))
    assert_size_stride(convolution_46, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_142, (800, ), (1, ))
    assert_size_stride(relu_47, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_47, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_145, (800, ), (1, ))
    assert_size_stride(relu_48, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_29, (8, 1344, 14, 14), (263424, 196, 14, 1))
    assert_size_stride(squeeze_148, (1344, ), (1, ))
    assert_size_stride(relu_49, (8, 1344, 14, 14), (263424, 196, 14, 1))
    assert_size_stride(convolution_49, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_151, (800, ), (1, ))
    assert_size_stride(relu_50, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_50, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_154, (800, ), (1, ))
    assert_size_stride(relu_51, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_31, (8, 1408, 14, 14), (275968, 196, 14, 1))
    assert_size_stride(squeeze_157, (1408, ), (1, ))
    assert_size_stride(relu_52, (8, 1408, 14, 14), (275968, 196, 14, 1))
    assert_size_stride(convolution_52, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_160, (800, ), (1, ))
    assert_size_stride(relu_53, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_53, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_163, (800, ), (1, ))
    assert_size_stride(relu_54, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_33, (8, 1472, 14, 14), (288512, 196, 14, 1))
    assert_size_stride(squeeze_166, (1472, ), (1, ))
    assert_size_stride(relu_55, (8, 1472, 14, 14), (288512, 196, 14, 1))
    assert_size_stride(convolution_55, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_169, (800, ), (1, ))
    assert_size_stride(relu_56, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_56, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_172, (800, ), (1, ))
    assert_size_stride(relu_57, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_35, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(squeeze_175, (1536, ), (1, ))
    assert_size_stride(relu_58, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(convolution_58, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_178, (800, ), (1, ))
    assert_size_stride(relu_59, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_59, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_181, (800, ), (1, ))
    assert_size_stride(relu_60, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_37, (8, 1600, 14, 14), (313600, 196, 14, 1))
    assert_size_stride(squeeze_184, (1600, ), (1, ))
    assert_size_stride(relu_61, (8, 1600, 14, 14), (313600, 196, 14, 1))
    assert_size_stride(convolution_61, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_187, (800, ), (1, ))
    assert_size_stride(relu_62, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_62, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_190, (800, ), (1, ))
    assert_size_stride(relu_63, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_39, (8, 1664, 14, 14), (326144, 196, 14, 1))
    assert_size_stride(squeeze_193, (1664, ), (1, ))
    assert_size_stride(relu_64, (8, 1664, 14, 14), (326144, 196, 14, 1))
    assert_size_stride(convolution_64, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_196, (800, ), (1, ))
    assert_size_stride(relu_65, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_65, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_199, (800, ), (1, ))
    assert_size_stride(relu_66, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_41, (8, 1728, 14, 14), (338688, 196, 14, 1))
    assert_size_stride(squeeze_202, (1728, ), (1, ))
    assert_size_stride(relu_67, (8, 1728, 14, 14), (338688, 196, 14, 1))
    assert_size_stride(convolution_67, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_205, (800, ), (1, ))
    assert_size_stride(relu_68, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_68, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_208, (800, ), (1, ))
    assert_size_stride(relu_69, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_43, (8, 1792, 14, 14), (351232, 196, 14, 1))
    assert_size_stride(squeeze_211, (1792, ), (1, ))
    assert_size_stride(relu_70, (8, 1792, 14, 14), (351232, 196, 14, 1))
    assert_size_stride(convolution_70, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_214, (800, ), (1, ))
    assert_size_stride(relu_71, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_71, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_217, (800, ), (1, ))
    assert_size_stride(relu_72, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_45, (8, 1856, 14, 14), (363776, 196, 14, 1))
    assert_size_stride(squeeze_220, (1856, ), (1, ))
    assert_size_stride(relu_73, (8, 1856, 14, 14), (363776, 196, 14, 1))
    assert_size_stride(convolution_73, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_223, (800, ), (1, ))
    assert_size_stride(relu_74, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_74, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_226, (800, ), (1, ))
    assert_size_stride(relu_75, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_47, (8, 1920, 14, 14), (376320, 196, 14, 1))
    assert_size_stride(squeeze_229, (1920, ), (1, ))
    assert_size_stride(relu_76, (8, 1920, 14, 14), (376320, 196, 14, 1))
    assert_size_stride(convolution_76, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_232, (800, ), (1, ))
    assert_size_stride(relu_77, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_77, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_235, (800, ), (1, ))
    assert_size_stride(relu_78, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_49, (8, 1984, 14, 14), (388864, 196, 14, 1))
    assert_size_stride(squeeze_238, (1984, ), (1, ))
    assert_size_stride(relu_79, (8, 1984, 14, 14), (388864, 196, 14, 1))
    assert_size_stride(convolution_79, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_241, (800, ), (1, ))
    assert_size_stride(relu_80, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_80, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_244, (800, ), (1, ))
    assert_size_stride(relu_81, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_51, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_247, (2048, ), (1, ))
    assert_size_stride(relu_82, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_82, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_250, (800, ), (1, ))
    assert_size_stride(relu_83, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_83, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_253, (800, ), (1, ))
    assert_size_stride(relu_84, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_53, (8, 2112, 14, 14), (413952, 196, 14, 1))
    assert_size_stride(squeeze_256, (2112, ), (1, ))
    assert_size_stride(relu_85, (8, 2112, 14, 14), (413952, 196, 14, 1))
    assert_size_stride(convolution_85, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_259, (800, ), (1, ))
    assert_size_stride(relu_86, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_86, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_262, (800, ), (1, ))
    assert_size_stride(relu_87, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_55, (8, 2176, 14, 14), (426496, 196, 14, 1))
    assert_size_stride(squeeze_265, (2176, ), (1, ))
    assert_size_stride(relu_88, (8, 2176, 14, 14), (426496, 196, 14, 1))
    assert_size_stride(convolution_88, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_268, (800, ), (1, ))
    assert_size_stride(relu_89, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_89, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_271, (800, ), (1, ))
    assert_size_stride(relu_90, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_57, (8, 2240, 14, 14), (439040, 196, 14, 1))
    assert_size_stride(squeeze_274, (2240, ), (1, ))
    assert_size_stride(relu_91, (8, 2240, 14, 14), (439040, 196, 14, 1))
    assert_size_stride(convolution_91, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_277, (800, ), (1, ))
    assert_size_stride(relu_92, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_92, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_280, (800, ), (1, ))
    assert_size_stride(relu_93, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_59, (8, 2304, 14, 14), (451584, 196, 14, 1))
    assert_size_stride(squeeze_283, (2304, ), (1, ))
    assert_size_stride(relu_94, (8, 2304, 14, 14), (451584, 196, 14, 1))
    assert_size_stride(convolution_94, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_286, (800, ), (1, ))
    assert_size_stride(relu_95, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_95, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_289, (800, ), (1, ))
    assert_size_stride(relu_96, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_61, (8, 2368, 14, 14), (464128, 196, 14, 1))
    assert_size_stride(squeeze_292, (2368, ), (1, ))
    assert_size_stride(relu_97, (8, 2368, 14, 14), (464128, 196, 14, 1))
    assert_size_stride(convolution_97, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_295, (800, ), (1, ))
    assert_size_stride(relu_98, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(convolution_98, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(squeeze_298, (800, ), (1, ))
    assert_size_stride(relu_99, (8, 800, 14, 14), (156800, 196, 14, 1))
    assert_size_stride(cat_63, (8, 2432, 14, 14), (476672, 196, 14, 1))
    assert_size_stride(squeeze_301, (2432, ), (1, ))
    assert_size_stride(relu_100, (8, 2432, 14, 14), (476672, 196, 14, 1))
    assert_size_stride(relu_101, (8, 2432, 14, 14), (476672, 196, 14, 1))
    assert_size_stride(convolution_101, (8, 1600, 14, 14), (313600, 196, 14, 1))
    assert_size_stride(squeeze_307, (1600, ), (1, ))
    assert_size_stride(relu_102, (8, 1600, 14, 14), (313600, 196, 14, 1))
    assert_size_stride(convolution_102, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(squeeze_310, (1600, ), (1, ))
    assert_size_stride(relu_103, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(cat_65, (8, 2432, 7, 7), (119168, 49, 7, 1))
    assert_size_stride(squeeze_313, (2432, ), (1, ))
    assert_size_stride(relu_104, (8, 2432, 7, 7), (119168, 49, 7, 1))
    assert_size_stride(convolution_104, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(squeeze_316, (1600, ), (1, ))
    assert_size_stride(relu_105, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(convolution_105, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(squeeze_319, (1600, ), (1, ))
    assert_size_stride(relu_106, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(cat_67, (8, 2560, 7, 7), (125440, 49, 7, 1))
    assert_size_stride(squeeze_322, (2560, ), (1, ))
    assert_size_stride(relu_107, (8, 2560, 7, 7), (125440, 49, 7, 1))
    assert_size_stride(convolution_107, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(squeeze_325, (1600, ), (1, ))
    assert_size_stride(relu_108, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(convolution_108, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(squeeze_328, (1600, ), (1, ))
    assert_size_stride(relu_109, (8, 1600, 7, 7), (78400, 49, 7, 1))
    assert_size_stride(cat_69, (8, 2688, 7, 7), (131712, 49, 7, 1))
    assert_size_stride(squeeze_331, (2688, ), (1, ))
    assert_size_stride(mean, (8, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(le, (8, 2688, 7, 7), (131712, 49, 7, 1))
    assert_size_stride(unsqueeze_446, (1, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_518, (1, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(unsqueeze_530, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_554, (1, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_590, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(unsqueeze_614, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_626, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_662, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_698, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_734, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_770, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_782, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_806, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_866, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_914, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_926, (1, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(unsqueeze_938, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_950, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_962, (1, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(unsqueeze_974, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_986, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_998, (1, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(unsqueeze_1010, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1022, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1034, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_1046, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1058, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1070, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(unsqueeze_1082, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1094, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1106, (1, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(unsqueeze_1118, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1130, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1142, (1, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(unsqueeze_1154, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1166, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1178, (1, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(unsqueeze_1190, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1202, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1214, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_1226, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1238, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1250, (1, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(unsqueeze_1262, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1274, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1286, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_1310, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1322, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1334, (1, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(unsqueeze_1346, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1358, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1370, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1382, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1394, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1406, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_1418, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1430, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1442, (1, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(unsqueeze_1454, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1466, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1478, (1, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(unsqueeze_1490, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1502, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1514, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1526, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1538, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1550, (1, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(unsqueeze_1562, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1574, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1586, (1, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(unsqueeze_1610, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1622, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1634, (1, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(unsqueeze_1646, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1658, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1670, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_1682, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1694, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1706, (1, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(unsqueeze_1718, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1730, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(sub_543, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(unsqueeze_1766, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_convolution_backward_0.run(tangents_1, buf0, 1000, 8, grid=grid(1000), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1 = aten.convolution_backward(reinterpret_tensor(tangents_1, (8, 1000, 1, 1), (1000, 1, 1, 1), 0), mean, primals_333, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_333
        del tangents_1
        buf2 = buf1[0]
        buf3 = buf1[1]
        del buf1
        buf4 = empty((2688, ), device='cuda', dtype=torch.float32)
        buf5 = empty((2688, ), device='cuda', dtype=torch.float32)
        buf7 = empty((2688, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf2, cat_69, unsqueeze_446, squeeze_331, buf4, buf5, buf7, 2688, 392, grid=grid(2688), stream=stream0)
        buf6 = empty((8, 2688, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_div_native_batch_norm_backward_threshold_backward_2.run(le, buf2, cat_69, unsqueeze_446, buf5, squeeze_331, buf4, primals_221, buf6, 1053696, grid=grid(1053696), stream=stream0)
        del buf2
        del buf5
        del cat_69
        del le
        del primals_221
        del squeeze_331
        del unsqueeze_446
        buf8 = empty((8, 2176, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_3.run(buf6, buf8, 852992, grid=grid(852992), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf9 = aten.convolution_backward(buf8, relu_109, primals_332, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_332
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf13 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf14 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(relu_109, buf10, convolution_108, unsqueeze_458, squeeze_328, buf12, buf13, buf14, 1600, 392, grid=grid(1600), stream=stream0)
        buf15 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5.run(buf15, relu_109, convolution_108, unsqueeze_458, buf13, squeeze_328, buf12, primals_219, 627200, grid=grid(627200), stream=stream0)
        del convolution_108
        del primals_219
        del relu_109
        del squeeze_328
        del unsqueeze_458
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf16 = aten.convolution_backward(buf15, relu_108, primals_331, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf15
        del primals_331
        buf17 = buf16[0]
        buf18 = buf16[1]
        del buf16
        buf19 = buf13; del buf13  # reuse
        buf20 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf21 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(relu_108, buf17, convolution_107, unsqueeze_470, squeeze_325, buf19, buf20, buf21, 1600, 392, grid=grid(1600), stream=stream0)
        buf22 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5.run(buf22, relu_108, convolution_107, unsqueeze_470, buf20, squeeze_325, buf19, primals_217, 627200, grid=grid(627200), stream=stream0)
        del convolution_107
        del primals_217
        del relu_108
        del squeeze_325
        del unsqueeze_470
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf23 = aten.convolution_backward(buf22, relu_107, primals_330, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf22
        del primals_330
        buf24 = buf23[0]
        buf25 = buf23[1]
        del buf23
        buf26 = empty((2560, ), device='cuda', dtype=torch.float32)
        buf27 = empty((2560, ), device='cuda', dtype=torch.float32)
        buf29 = empty((2560, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(relu_107, buf24, cat_67, unsqueeze_482, squeeze_322, buf26, buf27, buf29, 2560, 392, grid=grid(2560), stream=stream0)
        buf28 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_7.run(buf28, relu_107, cat_67, unsqueeze_482, buf27, squeeze_322, buf26, primals_215, 1003520, grid=grid(1003520), stream=stream0)
        del buf27
        del cat_67
        del primals_215
        del relu_107
        del squeeze_322
        del unsqueeze_482
        buf30 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_8.run(buf6, buf28, buf30, 852992, grid=grid(852992), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf31 = aten.convolution_backward(buf30, relu_106, primals_329, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_329
        buf32 = buf31[0]
        buf33 = buf31[1]
        del buf31
        buf34 = buf20; del buf20  # reuse
        buf35 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf36 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(relu_106, buf32, convolution_105, unsqueeze_494, squeeze_319, buf34, buf35, buf36, 1600, 392, grid=grid(1600), stream=stream0)
        buf37 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5.run(buf37, relu_106, convolution_105, unsqueeze_494, buf35, squeeze_319, buf34, primals_213, 627200, grid=grid(627200), stream=stream0)
        del convolution_105
        del primals_213
        del relu_106
        del squeeze_319
        del unsqueeze_494
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf38 = aten.convolution_backward(buf37, relu_105, primals_328, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf37
        del primals_328
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = buf35; del buf35  # reuse
        buf42 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf43 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(relu_105, buf39, convolution_104, unsqueeze_506, squeeze_316, buf41, buf42, buf43, 1600, 392, grid=grid(1600), stream=stream0)
        buf44 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5.run(buf44, relu_105, convolution_104, unsqueeze_506, buf42, squeeze_316, buf41, primals_211, 627200, grid=grid(627200), stream=stream0)
        del convolution_104
        del primals_211
        del relu_105
        del squeeze_316
        del unsqueeze_506
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf45 = aten.convolution_backward(buf44, relu_104, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf44
        del primals_327
        buf46 = buf45[0]
        buf47 = buf45[1]
        del buf45
        buf48 = empty((2432, ), device='cuda', dtype=torch.float32)
        buf49 = empty((2432, ), device='cuda', dtype=torch.float32)
        buf51 = empty((2432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(relu_104, buf46, cat_65, unsqueeze_518, squeeze_313, buf48, buf49, buf51, 2432, 392, grid=grid(2432), stream=stream0)
        buf50 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_10.run(buf50, relu_104, cat_65, unsqueeze_518, buf49, squeeze_313, buf48, primals_209, 953344, grid=grid(953344), stream=stream0)
        del cat_65
        del primals_209
        del relu_104
        del squeeze_313
        del unsqueeze_518
        buf52 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_11.run(buf6, buf28, buf50, buf52, 852992, grid=grid(852992), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf53 = aten.convolution_backward(buf52, relu_103, primals_326, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf52
        del primals_326
        buf54 = buf53[0]
        buf55 = buf53[1]
        del buf53
        buf56 = buf42; del buf42  # reuse
        buf57 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf58 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(relu_103, buf54, convolution_102, unsqueeze_530, squeeze_310, buf56, buf57, buf58, 1600, 392, grid=grid(1600), stream=stream0)
        buf59 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5.run(buf59, relu_103, convolution_102, unsqueeze_530, buf57, squeeze_310, buf56, primals_207, 627200, grid=grid(627200), stream=stream0)
        del convolution_102
        del primals_207
        del relu_103
        del squeeze_310
        del unsqueeze_530
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf60 = aten.convolution_backward(buf59, relu_102, primals_325, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf59
        del primals_325
        buf61 = buf60[0]
        buf62 = buf60[1]
        del buf60
        buf63 = buf57; del buf57  # reuse
        buf64 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf65 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_12.run(relu_102, buf61, convolution_101, unsqueeze_542, squeeze_307, buf63, buf64, buf65, 1600, 1568, grid=grid(1600), stream=stream0)
        buf66 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(buf66, relu_102, convolution_101, unsqueeze_542, buf64, squeeze_307, buf63, primals_205, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_101
        del primals_205
        del relu_102
        del squeeze_307
        del unsqueeze_542
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf67 = aten.convolution_backward(buf66, relu_101, primals_324, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf66
        del primals_324
        buf68 = buf67[0]
        buf69 = buf67[1]
        del buf67
        buf73 = empty((8, 2304, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_14.run(buf6, buf28, buf50, buf73, 903168, grid=grid(903168), stream=stream0)
        del buf50
        del buf6
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf74 = aten.convolution_backward(buf73, relu_100, primals_323, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf73
        del primals_323
        buf75 = buf74[0]
        buf70 = buf49; del buf49  # reuse
        buf71 = empty((2432, ), device='cuda', dtype=torch.float32)
        buf77 = empty((2432, ), device='cuda', dtype=torch.float32)
        buf78 = empty((2432, ), device='cuda', dtype=torch.float32)
        buf72 = empty((2432, ), device='cuda', dtype=torch.float32)
        buf79 = empty((2432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_15.run(relu_101, buf68, cat_63, unsqueeze_554, relu_100, buf75, squeeze_301, buf70, buf71, buf77, buf78, buf72, buf79, 2432, 1568, grid=grid(2432), stream=stream0)
        buf76 = buf74[1]
        del buf74
        buf80 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_16.run(buf80, relu_101, cat_63, unsqueeze_554, buf71, squeeze_301, buf70, primals_203, relu_100, buf75, buf78, buf77, primals_201, 3813376, grid=grid(3813376), stream=stream0)
        del buf71
        del buf75
        del buf78
        del cat_63
        del primals_201
        del primals_203
        del relu_100
        del relu_101
        del squeeze_301
        del unsqueeze_554
        buf81 = empty((8, 1088, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_17.run(buf80, buf81, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf82 = aten.convolution_backward(buf81, relu_99, primals_322, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_322
        buf83 = buf82[0]
        buf84 = buf82[1]
        del buf82
        buf85 = empty((800, ), device='cuda', dtype=torch.float32)
        buf86 = empty((800, ), device='cuda', dtype=torch.float32)
        buf87 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_99, buf83, convolution_98, unsqueeze_578, squeeze_298, buf85, buf86, buf87, 800, 1568, grid=grid(800), stream=stream0)
        buf88 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf88, relu_99, convolution_98, unsqueeze_578, buf86, squeeze_298, buf85, primals_199, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_98
        del primals_199
        del relu_99
        del squeeze_298
        del unsqueeze_578
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf89 = aten.convolution_backward(buf88, relu_98, primals_321, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf88
        del primals_321
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf92 = buf86; del buf86  # reuse
        buf93 = empty((800, ), device='cuda', dtype=torch.float32)
        buf94 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_98, buf90, convolution_97, unsqueeze_590, squeeze_295, buf92, buf93, buf94, 800, 1568, grid=grid(800), stream=stream0)
        buf95 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf95, relu_98, convolution_97, unsqueeze_590, buf93, squeeze_295, buf92, primals_197, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_97
        del primals_197
        del relu_98
        del squeeze_295
        del unsqueeze_590
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf96 = aten.convolution_backward(buf95, relu_97, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf95
        del primals_320
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf99 = empty((2368, ), device='cuda', dtype=torch.float32)
        buf100 = empty((2368, ), device='cuda', dtype=torch.float32)
        buf102 = empty((2368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_20.run(relu_97, buf97, cat_61, unsqueeze_602, squeeze_292, buf99, buf100, buf102, 2368, 1568, grid=grid(2368), stream=stream0)
        buf101 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_21.run(buf101, relu_97, cat_61, unsqueeze_602, buf100, squeeze_292, buf99, primals_195, 3713024, grid=grid(3713024), stream=stream0)
        del buf100
        del cat_61
        del primals_195
        del relu_97
        del squeeze_292
        del unsqueeze_602
        buf103 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_22.run(buf80, buf101, buf103, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf104 = aten.convolution_backward(buf103, relu_96, primals_319, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_319
        buf105 = buf104[0]
        buf106 = buf104[1]
        del buf104
        buf107 = buf93; del buf93  # reuse
        buf108 = empty((800, ), device='cuda', dtype=torch.float32)
        buf109 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_96, buf105, convolution_95, unsqueeze_614, squeeze_289, buf107, buf108, buf109, 800, 1568, grid=grid(800), stream=stream0)
        buf110 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf110, relu_96, convolution_95, unsqueeze_614, buf108, squeeze_289, buf107, primals_193, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_95
        del primals_193
        del relu_96
        del squeeze_289
        del unsqueeze_614
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf111 = aten.convolution_backward(buf110, relu_95, primals_318, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf110
        del primals_318
        buf112 = buf111[0]
        buf113 = buf111[1]
        del buf111
        buf114 = buf108; del buf108  # reuse
        buf115 = empty((800, ), device='cuda', dtype=torch.float32)
        buf116 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_95, buf112, convolution_94, unsqueeze_626, squeeze_286, buf114, buf115, buf116, 800, 1568, grid=grid(800), stream=stream0)
        buf117 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf117, relu_95, convolution_94, unsqueeze_626, buf115, squeeze_286, buf114, primals_191, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_94
        del primals_191
        del relu_95
        del squeeze_286
        del unsqueeze_626
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf118 = aten.convolution_backward(buf117, relu_94, primals_317, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf117
        del primals_317
        buf119 = buf118[0]
        buf120 = buf118[1]
        del buf118
        buf121 = empty((2304, ), device='cuda', dtype=torch.float32)
        buf122 = empty((2304, ), device='cuda', dtype=torch.float32)
        buf124 = empty((2304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(relu_94, buf119, cat_59, unsqueeze_638, squeeze_283, buf121, buf122, buf124, 2304, 1568, grid=grid(2304), stream=stream0)
        buf123 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_24.run(buf123, relu_94, cat_59, unsqueeze_638, buf122, squeeze_283, buf121, primals_189, 3612672, grid=grid(3612672), stream=stream0)
        del buf122
        del cat_59
        del primals_189
        del relu_94
        del squeeze_283
        del unsqueeze_638
        buf125 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_25.run(buf80, buf101, buf123, buf125, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf126 = aten.convolution_backward(buf125, relu_93, primals_316, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_316
        buf127 = buf126[0]
        buf128 = buf126[1]
        del buf126
        buf129 = buf115; del buf115  # reuse
        buf130 = empty((800, ), device='cuda', dtype=torch.float32)
        buf131 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_93, buf127, convolution_92, unsqueeze_650, squeeze_280, buf129, buf130, buf131, 800, 1568, grid=grid(800), stream=stream0)
        buf132 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf132, relu_93, convolution_92, unsqueeze_650, buf130, squeeze_280, buf129, primals_187, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_92
        del primals_187
        del relu_93
        del squeeze_280
        del unsqueeze_650
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf133 = aten.convolution_backward(buf132, relu_92, primals_315, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf132
        del primals_315
        buf134 = buf133[0]
        buf135 = buf133[1]
        del buf133
        buf136 = buf130; del buf130  # reuse
        buf137 = empty((800, ), device='cuda', dtype=torch.float32)
        buf138 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_92, buf134, convolution_91, unsqueeze_662, squeeze_277, buf136, buf137, buf138, 800, 1568, grid=grid(800), stream=stream0)
        buf139 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf139, relu_92, convolution_91, unsqueeze_662, buf137, squeeze_277, buf136, primals_185, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_91
        del primals_185
        del relu_92
        del squeeze_277
        del unsqueeze_662
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf140 = aten.convolution_backward(buf139, relu_91, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf139
        del primals_314
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf144 = empty((2240, ), device='cuda', dtype=torch.float32)
        buf146 = empty((2240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_91, buf141, cat_57, unsqueeze_674, squeeze_274, buf143, buf144, buf146, 2240, 1568, grid=grid(2240), stream=stream0)
        buf145 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf145, relu_91, cat_57, unsqueeze_674, buf144, squeeze_274, buf143, primals_183, 3512320, grid=grid(3512320), stream=stream0)
        del buf144
        del cat_57
        del primals_183
        del relu_91
        del squeeze_274
        del unsqueeze_674
        buf147 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_28.run(buf80, buf101, buf123, buf145, buf147, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf148 = aten.convolution_backward(buf147, relu_90, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_313
        buf149 = buf148[0]
        buf150 = buf148[1]
        del buf148
        buf151 = buf137; del buf137  # reuse
        buf152 = empty((800, ), device='cuda', dtype=torch.float32)
        buf153 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_90, buf149, convolution_89, unsqueeze_686, squeeze_271, buf151, buf152, buf153, 800, 1568, grid=grid(800), stream=stream0)
        buf154 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf154, relu_90, convolution_89, unsqueeze_686, buf152, squeeze_271, buf151, primals_181, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_89
        del primals_181
        del relu_90
        del squeeze_271
        del unsqueeze_686
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf155 = aten.convolution_backward(buf154, relu_89, primals_312, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf154
        del primals_312
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf158 = buf152; del buf152  # reuse
        buf159 = empty((800, ), device='cuda', dtype=torch.float32)
        buf160 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_89, buf156, convolution_88, unsqueeze_698, squeeze_268, buf158, buf159, buf160, 800, 1568, grid=grid(800), stream=stream0)
        buf161 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf161, relu_89, convolution_88, unsqueeze_698, buf159, squeeze_268, buf158, primals_179, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_88
        del primals_179
        del relu_89
        del squeeze_268
        del unsqueeze_698
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf162 = aten.convolution_backward(buf161, relu_88, primals_311, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf161
        del primals_311
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf165 = empty((2176, ), device='cuda', dtype=torch.float32)
        buf166 = empty((2176, ), device='cuda', dtype=torch.float32)
        buf168 = empty((2176, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(relu_88, buf163, cat_55, unsqueeze_710, squeeze_265, buf165, buf166, buf168, 2176, 1568, grid=grid(2176), stream=stream0)
        buf167 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_30.run(buf167, relu_88, cat_55, unsqueeze_710, buf166, squeeze_265, buf165, primals_177, 3411968, grid=grid(3411968), stream=stream0)
        del buf166
        del cat_55
        del primals_177
        del relu_88
        del squeeze_265
        del unsqueeze_710
        buf169 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf80, buf101, buf123, buf145, buf167, buf169, 1605632, grid=grid(1605632), stream=stream0)
        buf170 = empty((8, 1152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf80, buf101, buf123, buf145, buf167, buf170, 1806336, grid=grid(1806336), stream=stream0)
        del buf101
        del buf145
        del buf167
        del buf80
        buf171 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_33.run(buf170, buf169, buf171, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf172 = aten.convolution_backward(buf171, relu_87, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_310
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = buf159; del buf159  # reuse
        buf176 = empty((800, ), device='cuda', dtype=torch.float32)
        buf177 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_87, buf173, convolution_86, unsqueeze_722, squeeze_262, buf175, buf176, buf177, 800, 1568, grid=grid(800), stream=stream0)
        buf178 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf178, relu_87, convolution_86, unsqueeze_722, buf176, squeeze_262, buf175, primals_175, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_86
        del primals_175
        del relu_87
        del squeeze_262
        del unsqueeze_722
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf179 = aten.convolution_backward(buf178, relu_86, primals_309, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf178
        del primals_309
        buf180 = buf179[0]
        buf181 = buf179[1]
        del buf179
        buf182 = buf176; del buf176  # reuse
        buf183 = empty((800, ), device='cuda', dtype=torch.float32)
        buf184 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_86, buf180, convolution_85, unsqueeze_734, squeeze_259, buf182, buf183, buf184, 800, 1568, grid=grid(800), stream=stream0)
        buf185 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf185, relu_86, convolution_85, unsqueeze_734, buf183, squeeze_259, buf182, primals_173, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_85
        del primals_173
        del relu_86
        del squeeze_259
        del unsqueeze_734
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf186 = aten.convolution_backward(buf185, relu_85, primals_308, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf185
        del primals_308
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf189 = empty((2112, ), device='cuda', dtype=torch.float32)
        buf190 = empty((2112, ), device='cuda', dtype=torch.float32)
        buf192 = empty((2112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_85, buf187, cat_53, unsqueeze_746, squeeze_256, buf189, buf190, buf192, 2112, 1568, grid=grid(2112), stream=stream0)
        buf191 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_35.run(buf191, relu_85, cat_53, unsqueeze_746, buf190, squeeze_256, buf189, primals_171, 3311616, grid=grid(3311616), stream=stream0)
        del buf190
        del cat_53
        del primals_171
        del relu_85
        del squeeze_256
        del unsqueeze_746
        buf193 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_36.run(buf170, buf191, buf169, buf193, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf194 = aten.convolution_backward(buf193, relu_84, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_307
        buf195 = buf194[0]
        buf196 = buf194[1]
        del buf194
        buf197 = buf183; del buf183  # reuse
        buf198 = empty((800, ), device='cuda', dtype=torch.float32)
        buf199 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_84, buf195, convolution_83, unsqueeze_758, squeeze_253, buf197, buf198, buf199, 800, 1568, grid=grid(800), stream=stream0)
        buf200 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf200, relu_84, convolution_83, unsqueeze_758, buf198, squeeze_253, buf197, primals_169, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_83
        del primals_169
        del relu_84
        del squeeze_253
        del unsqueeze_758
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf201 = aten.convolution_backward(buf200, relu_83, primals_306, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf200
        del primals_306
        buf202 = buf201[0]
        buf203 = buf201[1]
        del buf201
        buf204 = buf198; del buf198  # reuse
        buf205 = empty((800, ), device='cuda', dtype=torch.float32)
        buf206 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_83, buf202, convolution_82, unsqueeze_770, squeeze_250, buf204, buf205, buf206, 800, 1568, grid=grid(800), stream=stream0)
        buf207 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf207, relu_83, convolution_82, unsqueeze_770, buf205, squeeze_250, buf204, primals_167, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_82
        del primals_167
        del relu_83
        del squeeze_250
        del unsqueeze_770
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf208 = aten.convolution_backward(buf207, relu_82, primals_305, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf207
        del primals_305
        buf209 = buf208[0]
        buf210 = buf208[1]
        del buf208
        buf211 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf212 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf214 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_82, buf209, cat_51, unsqueeze_782, squeeze_247, buf211, buf212, buf214, 2048, 1568, grid=grid(2048), stream=stream0)
        buf213 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_38.run(buf213, relu_82, cat_51, unsqueeze_782, buf212, squeeze_247, buf211, primals_165, 3211264, grid=grid(3211264), stream=stream0)
        del buf212
        del cat_51
        del primals_165
        del relu_82
        del squeeze_247
        del unsqueeze_782
        buf215 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_39.run(buf170, buf191, buf213, buf169, buf215, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf216 = aten.convolution_backward(buf215, relu_81, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_304
        buf217 = buf216[0]
        buf218 = buf216[1]
        del buf216
        buf219 = buf205; del buf205  # reuse
        buf220 = empty((800, ), device='cuda', dtype=torch.float32)
        buf221 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_81, buf217, convolution_80, unsqueeze_794, squeeze_244, buf219, buf220, buf221, 800, 1568, grid=grid(800), stream=stream0)
        buf222 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf222, relu_81, convolution_80, unsqueeze_794, buf220, squeeze_244, buf219, primals_163, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_80
        del primals_163
        del relu_81
        del squeeze_244
        del unsqueeze_794
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf223 = aten.convolution_backward(buf222, relu_80, primals_303, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf222
        del primals_303
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        buf226 = buf220; del buf220  # reuse
        buf227 = empty((800, ), device='cuda', dtype=torch.float32)
        buf228 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_80, buf224, convolution_79, unsqueeze_806, squeeze_241, buf226, buf227, buf228, 800, 1568, grid=grid(800), stream=stream0)
        buf229 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf229, relu_80, convolution_79, unsqueeze_806, buf227, squeeze_241, buf226, primals_161, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_79
        del primals_161
        del relu_80
        del squeeze_241
        del unsqueeze_806
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf230 = aten.convolution_backward(buf229, relu_79, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf229
        del primals_302
        buf231 = buf230[0]
        buf232 = buf230[1]
        del buf230
        buf233 = empty((1984, ), device='cuda', dtype=torch.float32)
        buf234 = empty((1984, ), device='cuda', dtype=torch.float32)
        buf236 = empty((1984, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(relu_79, buf231, cat_49, unsqueeze_818, squeeze_238, buf233, buf234, buf236, 1984, 1568, grid=grid(1984), stream=stream0)
        buf235 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_41.run(buf235, relu_79, cat_49, unsqueeze_818, buf234, squeeze_238, buf233, primals_159, 3110912, grid=grid(3110912), stream=stream0)
        del buf234
        del cat_49
        del primals_159
        del relu_79
        del squeeze_238
        del unsqueeze_818
        buf237 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_42.run(buf170, buf191, buf213, buf235, buf169, buf237, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf238 = aten.convolution_backward(buf237, relu_78, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_301
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        buf241 = buf227; del buf227  # reuse
        buf242 = empty((800, ), device='cuda', dtype=torch.float32)
        buf243 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_78, buf239, convolution_77, unsqueeze_830, squeeze_235, buf241, buf242, buf243, 800, 1568, grid=grid(800), stream=stream0)
        buf244 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf244, relu_78, convolution_77, unsqueeze_830, buf242, squeeze_235, buf241, primals_157, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_77
        del primals_157
        del relu_78
        del squeeze_235
        del unsqueeze_830
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf245 = aten.convolution_backward(buf244, relu_77, primals_300, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf244
        del primals_300
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = buf242; del buf242  # reuse
        buf249 = empty((800, ), device='cuda', dtype=torch.float32)
        buf250 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_77, buf246, convolution_76, unsqueeze_842, squeeze_232, buf248, buf249, buf250, 800, 1568, grid=grid(800), stream=stream0)
        buf251 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf251, relu_77, convolution_76, unsqueeze_842, buf249, squeeze_232, buf248, primals_155, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_76
        del primals_155
        del relu_77
        del squeeze_232
        del unsqueeze_842
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf252 = aten.convolution_backward(buf251, relu_76, primals_299, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf251
        del primals_299
        buf253 = buf252[0]
        buf254 = buf252[1]
        del buf252
        buf255 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf256 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf258 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(relu_76, buf253, cat_47, unsqueeze_854, squeeze_229, buf255, buf256, buf258, 1920, 1568, grid=grid(1920), stream=stream0)
        buf257 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_44.run(buf257, relu_76, cat_47, unsqueeze_854, buf256, squeeze_229, buf255, primals_153, 3010560, grid=grid(3010560), stream=stream0)
        del buf256
        del cat_47
        del primals_153
        del relu_76
        del squeeze_229
        del unsqueeze_854
        buf259 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf259, buf191, buf213, buf235, buf257, 1605632, grid=grid(1605632), stream=stream0)
        buf260 = empty((8, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(buf170, buf191, buf213, buf235, buf257, buf260, 1404928, grid=grid(1404928), stream=stream0)
        del buf191
        del buf235
        del buf257
        buf261 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_47.run(buf260, buf259, buf261, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf262 = aten.convolution_backward(buf261, relu_75, primals_298, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_298
        buf263 = buf262[0]
        buf264 = buf262[1]
        del buf262
        buf265 = buf249; del buf249  # reuse
        buf266 = empty((800, ), device='cuda', dtype=torch.float32)
        buf267 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_75, buf263, convolution_74, unsqueeze_866, squeeze_226, buf265, buf266, buf267, 800, 1568, grid=grid(800), stream=stream0)
        buf268 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf268, relu_75, convolution_74, unsqueeze_866, buf266, squeeze_226, buf265, primals_151, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_74
        del primals_151
        del relu_75
        del squeeze_226
        del unsqueeze_866
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf269 = aten.convolution_backward(buf268, relu_74, primals_297, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf268
        del primals_297
        buf270 = buf269[0]
        buf271 = buf269[1]
        del buf269
        buf272 = buf266; del buf266  # reuse
        buf273 = empty((800, ), device='cuda', dtype=torch.float32)
        buf274 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_74, buf270, convolution_73, unsqueeze_878, squeeze_223, buf272, buf273, buf274, 800, 1568, grid=grid(800), stream=stream0)
        buf275 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf275, relu_74, convolution_73, unsqueeze_878, buf273, squeeze_223, buf272, primals_149, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_73
        del primals_149
        del relu_74
        del squeeze_223
        del unsqueeze_878
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf276 = aten.convolution_backward(buf275, relu_73, primals_296, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf275
        del primals_296
        buf277 = buf276[0]
        buf278 = buf276[1]
        del buf276
        buf279 = empty((1856, ), device='cuda', dtype=torch.float32)
        buf280 = empty((1856, ), device='cuda', dtype=torch.float32)
        buf282 = empty((1856, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_73, buf277, cat_45, unsqueeze_890, squeeze_220, buf279, buf280, buf282, 1856, 1568, grid=grid(1856), stream=stream0)
        buf281 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_49.run(buf281, relu_73, cat_45, unsqueeze_890, buf280, squeeze_220, buf279, primals_147, 2910208, grid=grid(2910208), stream=stream0)
        del buf280
        del cat_45
        del primals_147
        del relu_73
        del squeeze_220
        del unsqueeze_890
        buf283 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_50.run(buf260, buf281, buf259, buf283, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf284 = aten.convolution_backward(buf283, relu_72, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_295
        buf285 = buf284[0]
        buf286 = buf284[1]
        del buf284
        buf287 = buf273; del buf273  # reuse
        buf288 = empty((800, ), device='cuda', dtype=torch.float32)
        buf289 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_72, buf285, convolution_71, unsqueeze_902, squeeze_217, buf287, buf288, buf289, 800, 1568, grid=grid(800), stream=stream0)
        buf290 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf290, relu_72, convolution_71, unsqueeze_902, buf288, squeeze_217, buf287, primals_145, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_71
        del primals_145
        del relu_72
        del squeeze_217
        del unsqueeze_902
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf291 = aten.convolution_backward(buf290, relu_71, primals_294, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf290
        del primals_294
        buf292 = buf291[0]
        buf293 = buf291[1]
        del buf291
        buf294 = buf288; del buf288  # reuse
        buf295 = empty((800, ), device='cuda', dtype=torch.float32)
        buf296 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_71, buf292, convolution_70, unsqueeze_914, squeeze_214, buf294, buf295, buf296, 800, 1568, grid=grid(800), stream=stream0)
        buf297 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf297, relu_71, convolution_70, unsqueeze_914, buf295, squeeze_214, buf294, primals_143, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_70
        del primals_143
        del relu_71
        del squeeze_214
        del unsqueeze_914
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf298 = aten.convolution_backward(buf297, relu_70, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf297
        del primals_293
        buf299 = buf298[0]
        buf300 = buf298[1]
        del buf298
        buf301 = empty((1792, ), device='cuda', dtype=torch.float32)
        buf302 = empty((1792, ), device='cuda', dtype=torch.float32)
        buf304 = empty((1792, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(relu_70, buf299, cat_43, unsqueeze_926, squeeze_211, buf301, buf302, buf304, 1792, 1568, grid=grid(1792), stream=stream0)
        buf303 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_52.run(buf303, relu_70, cat_43, unsqueeze_926, buf302, squeeze_211, buf301, primals_141, 2809856, grid=grid(2809856), stream=stream0)
        del buf302
        del cat_43
        del primals_141
        del relu_70
        del squeeze_211
        del unsqueeze_926
        buf305 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_53.run(buf260, buf281, buf303, buf259, buf305, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf306 = aten.convolution_backward(buf305, relu_69, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_292
        buf307 = buf306[0]
        buf308 = buf306[1]
        del buf306
        buf309 = buf295; del buf295  # reuse
        buf310 = empty((800, ), device='cuda', dtype=torch.float32)
        buf311 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_69, buf307, convolution_68, unsqueeze_938, squeeze_208, buf309, buf310, buf311, 800, 1568, grid=grid(800), stream=stream0)
        buf312 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf312, relu_69, convolution_68, unsqueeze_938, buf310, squeeze_208, buf309, primals_139, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_68
        del primals_139
        del relu_69
        del squeeze_208
        del unsqueeze_938
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf313 = aten.convolution_backward(buf312, relu_68, primals_291, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf312
        del primals_291
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = buf310; del buf310  # reuse
        buf317 = empty((800, ), device='cuda', dtype=torch.float32)
        buf318 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_68, buf314, convolution_67, unsqueeze_950, squeeze_205, buf316, buf317, buf318, 800, 1568, grid=grid(800), stream=stream0)
        buf319 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf319, relu_68, convolution_67, unsqueeze_950, buf317, squeeze_205, buf316, primals_137, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_67
        del primals_137
        del relu_68
        del squeeze_205
        del unsqueeze_950
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf320 = aten.convolution_backward(buf319, relu_67, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf319
        del primals_290
        buf321 = buf320[0]
        buf322 = buf320[1]
        del buf320
        buf323 = empty((1728, ), device='cuda', dtype=torch.float32)
        buf324 = empty((1728, ), device='cuda', dtype=torch.float32)
        buf326 = empty((1728, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(relu_67, buf321, cat_41, unsqueeze_962, squeeze_202, buf323, buf324, buf326, 1728, 1568, grid=grid(1728), stream=stream0)
        buf325 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_55.run(buf325, relu_67, cat_41, unsqueeze_962, buf324, squeeze_202, buf323, primals_135, 2709504, grid=grid(2709504), stream=stream0)
        del buf324
        del cat_41
        del primals_135
        del relu_67
        del squeeze_202
        del unsqueeze_962
        buf327 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_56.run(buf260, buf281, buf303, buf325, buf259, buf327, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf328 = aten.convolution_backward(buf327, relu_66, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_289
        buf329 = buf328[0]
        buf330 = buf328[1]
        del buf328
        buf331 = buf317; del buf317  # reuse
        buf332 = empty((800, ), device='cuda', dtype=torch.float32)
        buf333 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_66, buf329, convolution_65, unsqueeze_974, squeeze_199, buf331, buf332, buf333, 800, 1568, grid=grid(800), stream=stream0)
        buf334 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf334, relu_66, convolution_65, unsqueeze_974, buf332, squeeze_199, buf331, primals_133, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_65
        del primals_133
        del relu_66
        del squeeze_199
        del unsqueeze_974
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf335 = aten.convolution_backward(buf334, relu_65, primals_288, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf334
        del primals_288
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        buf338 = buf332; del buf332  # reuse
        buf339 = empty((800, ), device='cuda', dtype=torch.float32)
        buf340 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_65, buf336, convolution_64, unsqueeze_986, squeeze_196, buf338, buf339, buf340, 800, 1568, grid=grid(800), stream=stream0)
        buf341 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf341, relu_65, convolution_64, unsqueeze_986, buf339, squeeze_196, buf338, primals_131, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_64
        del primals_131
        del relu_65
        del squeeze_196
        del unsqueeze_986
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf342 = aten.convolution_backward(buf341, relu_64, primals_287, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf341
        del primals_287
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        buf345 = empty((1664, ), device='cuda', dtype=torch.float32)
        buf346 = empty((1664, ), device='cuda', dtype=torch.float32)
        buf348 = empty((1664, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_64, buf343, cat_39, unsqueeze_998, squeeze_193, buf345, buf346, buf348, 1664, 1568, grid=grid(1664), stream=stream0)
        buf347 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_58.run(buf347, relu_64, cat_39, unsqueeze_998, buf346, squeeze_193, buf345, primals_129, 2609152, grid=grid(2609152), stream=stream0)
        del buf346
        del cat_39
        del primals_129
        del relu_64
        del squeeze_193
        del unsqueeze_998
        buf349 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf349, buf281, buf303, buf325, buf347, 1605632, grid=grid(1605632), stream=stream0)
        buf350 = reinterpret_tensor(buf28, (8, 640, 14, 14), (125440, 196, 14, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_60.run(buf260, buf281, buf303, buf325, buf347, buf350, 1003520, grid=grid(1003520), stream=stream0)
        del buf260
        del buf281
        del buf303
        del buf325
        del buf347
        buf351 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_61.run(buf350, buf349, buf351, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf352 = aten.convolution_backward(buf351, relu_63, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_286
        buf353 = buf352[0]
        buf354 = buf352[1]
        del buf352
        buf355 = buf339; del buf339  # reuse
        buf356 = empty((800, ), device='cuda', dtype=torch.float32)
        buf357 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_63, buf353, convolution_62, unsqueeze_1010, squeeze_190, buf355, buf356, buf357, 800, 1568, grid=grid(800), stream=stream0)
        buf358 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf358, relu_63, convolution_62, unsqueeze_1010, buf356, squeeze_190, buf355, primals_127, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_62
        del primals_127
        del relu_63
        del squeeze_190
        del unsqueeze_1010
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf359 = aten.convolution_backward(buf358, relu_62, primals_285, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf358
        del primals_285
        buf360 = buf359[0]
        buf361 = buf359[1]
        del buf359
        buf362 = buf356; del buf356  # reuse
        buf363 = empty((800, ), device='cuda', dtype=torch.float32)
        buf364 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_62, buf360, convolution_61, unsqueeze_1022, squeeze_187, buf362, buf363, buf364, 800, 1568, grid=grid(800), stream=stream0)
        buf365 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf365, relu_62, convolution_61, unsqueeze_1022, buf363, squeeze_187, buf362, primals_125, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_61
        del primals_125
        del relu_62
        del squeeze_187
        del unsqueeze_1022
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf366 = aten.convolution_backward(buf365, relu_61, primals_284, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf365
        del primals_284
        buf367 = buf366[0]
        buf368 = buf366[1]
        del buf366
        buf369 = buf64; del buf64  # reuse
        buf370 = empty((1600, ), device='cuda', dtype=torch.float32)
        buf372 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_12.run(relu_61, buf367, cat_37, unsqueeze_1034, squeeze_184, buf369, buf370, buf372, 1600, 1568, grid=grid(1600), stream=stream0)
        buf371 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(buf371, relu_61, cat_37, unsqueeze_1034, buf370, squeeze_184, buf369, primals_123, 2508800, grid=grid(2508800), stream=stream0)
        del buf370
        del cat_37
        del primals_123
        del relu_61
        del squeeze_184
        del unsqueeze_1034
        buf373 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_62.run(buf350, buf371, buf349, buf373, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf374 = aten.convolution_backward(buf373, relu_60, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_283
        buf375 = buf374[0]
        buf376 = buf374[1]
        del buf374
        buf377 = buf363; del buf363  # reuse
        buf378 = empty((800, ), device='cuda', dtype=torch.float32)
        buf379 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_60, buf375, convolution_59, unsqueeze_1046, squeeze_181, buf377, buf378, buf379, 800, 1568, grid=grid(800), stream=stream0)
        buf380 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf380, relu_60, convolution_59, unsqueeze_1046, buf378, squeeze_181, buf377, primals_121, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_59
        del primals_121
        del relu_60
        del squeeze_181
        del unsqueeze_1046
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf381 = aten.convolution_backward(buf380, relu_59, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf380
        del primals_282
        buf382 = buf381[0]
        buf383 = buf381[1]
        del buf381
        buf384 = buf378; del buf378  # reuse
        buf385 = empty((800, ), device='cuda', dtype=torch.float32)
        buf386 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_59, buf382, convolution_58, unsqueeze_1058, squeeze_178, buf384, buf385, buf386, 800, 1568, grid=grid(800), stream=stream0)
        buf387 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf387, relu_59, convolution_58, unsqueeze_1058, buf385, squeeze_178, buf384, primals_119, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_58
        del primals_119
        del relu_59
        del squeeze_178
        del unsqueeze_1058
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf388 = aten.convolution_backward(buf387, relu_58, primals_281, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf387
        del primals_281
        buf389 = buf388[0]
        buf390 = buf388[1]
        del buf388
        buf391 = empty((1536, ), device='cuda', dtype=torch.float32)
        buf392 = empty((1536, ), device='cuda', dtype=torch.float32)
        buf394 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_63.run(relu_58, buf389, cat_35, unsqueeze_1070, squeeze_175, buf391, buf392, buf394, 1536, 1568, grid=grid(1536), stream=stream0)
        buf393 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_64.run(buf393, relu_58, cat_35, unsqueeze_1070, buf392, squeeze_175, buf391, primals_117, 2408448, grid=grid(2408448), stream=stream0)
        del buf392
        del cat_35
        del primals_117
        del relu_58
        del squeeze_175
        del unsqueeze_1070
        buf395 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_65.run(buf350, buf371, buf393, buf349, buf395, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf396 = aten.convolution_backward(buf395, relu_57, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_280
        buf397 = buf396[0]
        buf398 = buf396[1]
        del buf396
        buf399 = buf385; del buf385  # reuse
        buf400 = empty((800, ), device='cuda', dtype=torch.float32)
        buf401 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_57, buf397, convolution_56, unsqueeze_1082, squeeze_172, buf399, buf400, buf401, 800, 1568, grid=grid(800), stream=stream0)
        buf402 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf402, relu_57, convolution_56, unsqueeze_1082, buf400, squeeze_172, buf399, primals_115, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_56
        del primals_115
        del relu_57
        del squeeze_172
        del unsqueeze_1082
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf403 = aten.convolution_backward(buf402, relu_56, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf402
        del primals_279
        buf404 = buf403[0]
        buf405 = buf403[1]
        del buf403
        buf406 = buf400; del buf400  # reuse
        buf407 = empty((800, ), device='cuda', dtype=torch.float32)
        buf408 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_56, buf404, convolution_55, unsqueeze_1094, squeeze_169, buf406, buf407, buf408, 800, 1568, grid=grid(800), stream=stream0)
        buf409 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf409, relu_56, convolution_55, unsqueeze_1094, buf407, squeeze_169, buf406, primals_113, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_55
        del primals_113
        del relu_56
        del squeeze_169
        del unsqueeze_1094
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf410 = aten.convolution_backward(buf409, relu_55, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf409
        del primals_278
        buf411 = buf410[0]
        buf412 = buf410[1]
        del buf410
        buf413 = empty((1472, ), device='cuda', dtype=torch.float32)
        buf414 = empty((1472, ), device='cuda', dtype=torch.float32)
        buf416 = empty((1472, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_55, buf411, cat_33, unsqueeze_1106, squeeze_166, buf413, buf414, buf416, 1472, 1568, grid=grid(1472), stream=stream0)
        buf415 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_67.run(buf415, relu_55, cat_33, unsqueeze_1106, buf414, squeeze_166, buf413, primals_111, 2308096, grid=grid(2308096), stream=stream0)
        del buf414
        del cat_33
        del primals_111
        del relu_55
        del squeeze_166
        del unsqueeze_1106
        buf417 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_68.run(buf350, buf371, buf393, buf415, buf349, buf417, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf418 = aten.convolution_backward(buf417, relu_54, primals_277, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_277
        buf419 = buf418[0]
        buf420 = buf418[1]
        del buf418
        buf421 = buf407; del buf407  # reuse
        buf422 = empty((800, ), device='cuda', dtype=torch.float32)
        buf423 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_54, buf419, convolution_53, unsqueeze_1118, squeeze_163, buf421, buf422, buf423, 800, 1568, grid=grid(800), stream=stream0)
        buf424 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf424, relu_54, convolution_53, unsqueeze_1118, buf422, squeeze_163, buf421, primals_109, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_53
        del primals_109
        del relu_54
        del squeeze_163
        del unsqueeze_1118
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf425 = aten.convolution_backward(buf424, relu_53, primals_276, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf424
        del primals_276
        buf426 = buf425[0]
        buf427 = buf425[1]
        del buf425
        buf428 = buf422; del buf422  # reuse
        buf429 = empty((800, ), device='cuda', dtype=torch.float32)
        buf430 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_53, buf426, convolution_52, unsqueeze_1130, squeeze_160, buf428, buf429, buf430, 800, 1568, grid=grid(800), stream=stream0)
        buf431 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf431, relu_53, convolution_52, unsqueeze_1130, buf429, squeeze_160, buf428, primals_107, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_52
        del primals_107
        del relu_53
        del squeeze_160
        del unsqueeze_1130
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf432 = aten.convolution_backward(buf431, relu_52, primals_275, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf431
        del primals_275
        buf433 = buf432[0]
        buf434 = buf432[1]
        del buf432
        buf435 = empty((1408, ), device='cuda', dtype=torch.float32)
        buf436 = empty((1408, ), device='cuda', dtype=torch.float32)
        buf438 = empty((1408, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(relu_52, buf433, cat_31, unsqueeze_1142, squeeze_157, buf435, buf436, buf438, 1408, 1568, grid=grid(1408), stream=stream0)
        buf437 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_70.run(buf437, relu_52, cat_31, unsqueeze_1142, buf436, squeeze_157, buf435, primals_105, 2207744, grid=grid(2207744), stream=stream0)
        del buf436
        del cat_31
        del primals_105
        del relu_52
        del squeeze_157
        del unsqueeze_1142
        buf439 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_71.run(buf439, buf371, buf393, buf415, buf437, 1605632, grid=grid(1605632), stream=stream0)
        buf440 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_72.run(buf350, buf371, buf393, buf415, buf437, buf440, 602112, grid=grid(602112), stream=stream0)
        del buf350
        del buf371
        del buf415
        del buf437
        buf441 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_73.run(buf440, buf439, buf441, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf442 = aten.convolution_backward(buf441, relu_51, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_274
        buf443 = buf442[0]
        buf444 = buf442[1]
        del buf442
        buf445 = buf429; del buf429  # reuse
        buf446 = empty((800, ), device='cuda', dtype=torch.float32)
        buf447 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_51, buf443, convolution_50, unsqueeze_1154, squeeze_154, buf445, buf446, buf447, 800, 1568, grid=grid(800), stream=stream0)
        buf448 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf448, relu_51, convolution_50, unsqueeze_1154, buf446, squeeze_154, buf445, primals_103, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_50
        del primals_103
        del relu_51
        del squeeze_154
        del unsqueeze_1154
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf449 = aten.convolution_backward(buf448, relu_50, primals_273, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf448
        del primals_273
        buf450 = buf449[0]
        buf451 = buf449[1]
        del buf449
        buf452 = buf446; del buf446  # reuse
        buf453 = empty((800, ), device='cuda', dtype=torch.float32)
        buf454 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_50, buf450, convolution_49, unsqueeze_1166, squeeze_151, buf452, buf453, buf454, 800, 1568, grid=grid(800), stream=stream0)
        buf455 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf455, relu_50, convolution_49, unsqueeze_1166, buf453, squeeze_151, buf452, primals_101, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_49
        del primals_101
        del relu_50
        del squeeze_151
        del unsqueeze_1166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf456 = aten.convolution_backward(buf455, relu_49, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf455
        del primals_272
        buf457 = buf456[0]
        buf458 = buf456[1]
        del buf456
        buf459 = empty((1344, ), device='cuda', dtype=torch.float32)
        buf460 = empty((1344, ), device='cuda', dtype=torch.float32)
        buf462 = empty((1344, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_49, buf457, cat_29, unsqueeze_1178, squeeze_148, buf459, buf460, buf462, 1344, 1568, grid=grid(1344), stream=stream0)
        buf461 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_75.run(buf461, relu_49, cat_29, unsqueeze_1178, buf460, squeeze_148, buf459, primals_99, 2107392, grid=grid(2107392), stream=stream0)
        del buf460
        del cat_29
        del primals_99
        del relu_49
        del squeeze_148
        del unsqueeze_1178
        buf463 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_76.run(buf440, buf461, buf439, buf463, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf464 = aten.convolution_backward(buf463, relu_48, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_271
        buf465 = buf464[0]
        buf466 = buf464[1]
        del buf464
        buf467 = buf453; del buf453  # reuse
        buf468 = empty((800, ), device='cuda', dtype=torch.float32)
        buf469 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_48, buf465, convolution_47, unsqueeze_1190, squeeze_145, buf467, buf468, buf469, 800, 1568, grid=grid(800), stream=stream0)
        buf470 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf470, relu_48, convolution_47, unsqueeze_1190, buf468, squeeze_145, buf467, primals_97, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_47
        del primals_97
        del relu_48
        del squeeze_145
        del unsqueeze_1190
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf471 = aten.convolution_backward(buf470, relu_47, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf470
        del primals_270
        buf472 = buf471[0]
        buf473 = buf471[1]
        del buf471
        buf474 = buf468; del buf468  # reuse
        buf475 = empty((800, ), device='cuda', dtype=torch.float32)
        buf476 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_47, buf472, convolution_46, unsqueeze_1202, squeeze_142, buf474, buf475, buf476, 800, 1568, grid=grid(800), stream=stream0)
        buf477 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf477, relu_47, convolution_46, unsqueeze_1202, buf475, squeeze_142, buf474, primals_95, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_46
        del primals_95
        del relu_47
        del squeeze_142
        del unsqueeze_1202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf478 = aten.convolution_backward(buf477, relu_46, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf477
        del primals_269
        buf479 = buf478[0]
        buf480 = buf478[1]
        del buf478
        buf481 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf482 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf484 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_77.run(relu_46, buf479, cat_27, unsqueeze_1214, squeeze_139, buf481, buf482, buf484, 1280, 1568, grid=grid(1280), stream=stream0)
        buf483 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_78.run(buf483, relu_46, cat_27, unsqueeze_1214, buf482, squeeze_139, buf481, primals_93, 2007040, grid=grid(2007040), stream=stream0)
        del buf482
        del cat_27
        del primals_93
        del relu_46
        del squeeze_139
        del unsqueeze_1214
        buf485 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_79.run(buf440, buf461, buf483, buf439, buf485, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf486 = aten.convolution_backward(buf485, relu_45, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_268
        buf487 = buf486[0]
        buf488 = buf486[1]
        del buf486
        buf489 = buf475; del buf475  # reuse
        buf490 = empty((800, ), device='cuda', dtype=torch.float32)
        buf491 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_45, buf487, convolution_44, unsqueeze_1226, squeeze_136, buf489, buf490, buf491, 800, 1568, grid=grid(800), stream=stream0)
        buf492 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf492, relu_45, convolution_44, unsqueeze_1226, buf490, squeeze_136, buf489, primals_91, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_44
        del primals_91
        del relu_45
        del squeeze_136
        del unsqueeze_1226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf493 = aten.convolution_backward(buf492, relu_44, primals_267, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf492
        del primals_267
        buf494 = buf493[0]
        buf495 = buf493[1]
        del buf493
        buf496 = buf490; del buf490  # reuse
        buf497 = empty((800, ), device='cuda', dtype=torch.float32)
        buf498 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_44, buf494, convolution_43, unsqueeze_1238, squeeze_133, buf496, buf497, buf498, 800, 1568, grid=grid(800), stream=stream0)
        buf499 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf499, relu_44, convolution_43, unsqueeze_1238, buf497, squeeze_133, buf496, primals_89, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_43
        del primals_89
        del relu_44
        del squeeze_133
        del unsqueeze_1238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf500 = aten.convolution_backward(buf499, relu_43, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf499
        del primals_266
        buf501 = buf500[0]
        buf502 = buf500[1]
        del buf500
        buf503 = empty((1216, ), device='cuda', dtype=torch.float32)
        buf504 = empty((1216, ), device='cuda', dtype=torch.float32)
        buf506 = empty((1216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_43, buf501, cat_25, unsqueeze_1250, squeeze_130, buf503, buf504, buf506, 1216, 1568, grid=grid(1216), stream=stream0)
        buf505 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_81.run(buf505, relu_43, cat_25, unsqueeze_1250, buf504, squeeze_130, buf503, primals_87, 1906688, grid=grid(1906688), stream=stream0)
        del buf504
        del cat_25
        del primals_87
        del relu_43
        del squeeze_130
        del unsqueeze_1250
        buf507 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_82.run(buf440, buf461, buf483, buf505, buf439, buf507, 1705984, grid=grid(1705984), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf508 = aten.convolution_backward(buf507, relu_42, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf507
        del primals_265
        buf509 = buf508[0]
        buf510 = buf508[1]
        del buf508
        buf511 = buf497; del buf497  # reuse
        buf512 = empty((800, ), device='cuda', dtype=torch.float32)
        buf513 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_42, buf509, convolution_41, unsqueeze_1262, squeeze_127, buf511, buf512, buf513, 800, 1568, grid=grid(800), stream=stream0)
        buf514 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf514, relu_42, convolution_41, unsqueeze_1262, buf512, squeeze_127, buf511, primals_85, 1254400, grid=grid(1254400), stream=stream0)
        del convolution_41
        del primals_85
        del relu_42
        del squeeze_127
        del unsqueeze_1262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf515 = aten.convolution_backward(buf514, relu_41, primals_264, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf514
        del primals_264
        buf516 = buf515[0]
        buf517 = buf515[1]
        del buf515
        buf518 = buf512; del buf512  # reuse
        buf519 = empty((800, ), device='cuda', dtype=torch.float32)
        buf520 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_83.run(relu_41, buf516, convolution_40, unsqueeze_1274, squeeze_124, buf518, buf519, buf520, 800, 6272, grid=grid(800), stream=stream0)
        buf521 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84.run(buf521, relu_41, convolution_40, unsqueeze_1274, buf519, squeeze_124, buf518, primals_83, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_40
        del primals_83
        del relu_41
        del squeeze_124
        del unsqueeze_1274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf522 = aten.convolution_backward(buf521, relu_40, primals_263, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf521
        del primals_263
        buf523 = buf522[0]
        buf524 = buf522[1]
        del buf522
        buf528 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_85.run(buf440, buf461, buf483, buf505, buf439, buf528, 1806336, grid=grid(1806336), stream=stream0)
        del buf439
        del buf440
        del buf461
        del buf483
        del buf505
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf529 = aten.convolution_backward(buf528, relu_39, primals_262, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf528
        del primals_262
        buf530 = buf529[0]
        buf525 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf526 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf532 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf533 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf527 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf534 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(relu_40, buf523, cat_23, unsqueeze_1286, relu_39, buf530, squeeze_118, buf525, buf526, buf532, buf533, buf527, buf534, 1152, 6272, grid=grid(1152), stream=stream0)
        buf531 = buf529[1]
        del buf529
        buf535 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_87.run(buf535, relu_40, cat_23, unsqueeze_1286, buf526, squeeze_118, buf525, primals_81, relu_39, buf530, buf533, buf532, primals_79, 7225344, grid=grid(7225344), stream=stream0)
        del buf526
        del buf530
        del buf533
        del cat_23
        del primals_79
        del primals_81
        del relu_39
        del relu_40
        del squeeze_118
        del unsqueeze_1286
        buf536 = reinterpret_tensor(buf123, (8, 576, 28, 28), (451584, 784, 28, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_88.run(buf535, buf536, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf537 = aten.convolution_backward(buf536, relu_38, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_261
        buf538 = buf537[0]
        buf539 = buf537[1]
        del buf537
        buf540 = empty((400, ), device='cuda', dtype=torch.float32)
        buf541 = empty((400, ), device='cuda', dtype=torch.float32)
        buf542 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_38, buf538, convolution_37, unsqueeze_1310, squeeze_115, buf540, buf541, buf542, 400, 6272, grid=grid(400), stream=stream0)
        buf543 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf543, relu_38, convolution_37, unsqueeze_1310, buf541, squeeze_115, buf540, primals_77, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_37
        del primals_77
        del relu_38
        del squeeze_115
        del unsqueeze_1310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf544 = aten.convolution_backward(buf543, relu_37, primals_260, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf543
        del primals_260
        buf545 = buf544[0]
        buf546 = buf544[1]
        del buf544
        buf547 = buf541; del buf541  # reuse
        buf548 = empty((400, ), device='cuda', dtype=torch.float32)
        buf549 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_37, buf545, convolution_36, unsqueeze_1322, squeeze_112, buf547, buf548, buf549, 400, 6272, grid=grid(400), stream=stream0)
        buf550 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf550, relu_37, convolution_36, unsqueeze_1322, buf548, squeeze_112, buf547, primals_75, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_36
        del primals_75
        del relu_37
        del squeeze_112
        del unsqueeze_1322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf551 = aten.convolution_backward(buf550, relu_36, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf550
        del primals_259
        buf552 = buf551[0]
        buf553 = buf551[1]
        del buf551
        buf554 = empty((1088, ), device='cuda', dtype=torch.float32)
        buf555 = empty((1088, ), device='cuda', dtype=torch.float32)
        buf557 = empty((1088, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(relu_36, buf552, cat_21, unsqueeze_1334, squeeze_109, buf554, buf555, buf557, 1088, 6272, grid=grid(1088), stream=stream0)
        buf556 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_92.run(buf556, relu_36, cat_21, unsqueeze_1334, buf555, squeeze_109, buf554, primals_73, 6823936, grid=grid(6823936), stream=stream0)
        del buf555
        del cat_21
        del primals_73
        del relu_36
        del squeeze_109
        del unsqueeze_1334
        buf558 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_93.run(buf535, buf556, buf558, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf559 = aten.convolution_backward(buf558, relu_35, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_258
        buf560 = buf559[0]
        buf561 = buf559[1]
        del buf559
        buf562 = buf548; del buf548  # reuse
        buf563 = empty((400, ), device='cuda', dtype=torch.float32)
        buf564 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_35, buf560, convolution_34, unsqueeze_1346, squeeze_106, buf562, buf563, buf564, 400, 6272, grid=grid(400), stream=stream0)
        buf565 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf565, relu_35, convolution_34, unsqueeze_1346, buf563, squeeze_106, buf562, primals_71, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_34
        del primals_71
        del relu_35
        del squeeze_106
        del unsqueeze_1346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf566 = aten.convolution_backward(buf565, relu_34, primals_257, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf565
        del primals_257
        buf567 = buf566[0]
        buf568 = buf566[1]
        del buf566
        buf569 = buf563; del buf563  # reuse
        buf570 = empty((400, ), device='cuda', dtype=torch.float32)
        buf571 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_34, buf567, convolution_33, unsqueeze_1358, squeeze_103, buf569, buf570, buf571, 400, 6272, grid=grid(400), stream=stream0)
        buf572 = buf567; del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf572, relu_34, convolution_33, unsqueeze_1358, buf570, squeeze_103, buf569, primals_69, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_33
        del primals_69
        del relu_34
        del squeeze_103
        del unsqueeze_1358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf573 = aten.convolution_backward(buf572, relu_33, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf572
        del primals_256
        buf574 = buf573[0]
        buf575 = buf573[1]
        del buf573
        buf576 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf577 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf579 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_94.run(relu_33, buf574, cat_19, unsqueeze_1370, squeeze_100, buf576, buf577, buf579, 1024, 6272, grid=grid(1024), stream=stream0)
        buf578 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_95.run(buf578, relu_33, cat_19, unsqueeze_1370, buf577, squeeze_100, buf576, primals_67, 6422528, grid=grid(6422528), stream=stream0)
        del buf577
        del cat_19
        del primals_67
        del relu_33
        del squeeze_100
        del unsqueeze_1370
        buf580 = buf558; del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_96.run(buf535, buf556, buf578, buf580, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf581 = aten.convolution_backward(buf580, relu_32, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_255
        buf582 = buf581[0]
        buf583 = buf581[1]
        del buf581
        buf584 = buf570; del buf570  # reuse
        buf585 = empty((400, ), device='cuda', dtype=torch.float32)
        buf586 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_32, buf582, convolution_31, unsqueeze_1382, squeeze_97, buf584, buf585, buf586, 400, 6272, grid=grid(400), stream=stream0)
        buf587 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf587, relu_32, convolution_31, unsqueeze_1382, buf585, squeeze_97, buf584, primals_65, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_31
        del primals_65
        del relu_32
        del squeeze_97
        del unsqueeze_1382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf588 = aten.convolution_backward(buf587, relu_31, primals_254, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf587
        del primals_254
        buf589 = buf588[0]
        buf590 = buf588[1]
        del buf588
        buf591 = buf585; del buf585  # reuse
        buf592 = empty((400, ), device='cuda', dtype=torch.float32)
        buf593 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_31, buf589, convolution_30, unsqueeze_1394, squeeze_94, buf591, buf592, buf593, 400, 6272, grid=grid(400), stream=stream0)
        buf594 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf594, relu_31, convolution_30, unsqueeze_1394, buf592, squeeze_94, buf591, primals_63, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_30
        del primals_63
        del relu_31
        del squeeze_94
        del unsqueeze_1394
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf595 = aten.convolution_backward(buf594, relu_30, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf594
        del primals_253
        buf596 = buf595[0]
        buf597 = buf595[1]
        del buf595
        buf598 = empty((960, ), device='cuda', dtype=torch.float32)
        buf599 = empty((960, ), device='cuda', dtype=torch.float32)
        buf601 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(relu_30, buf596, cat_17, unsqueeze_1406, squeeze_91, buf598, buf599, buf601, 960, 6272, grid=grid(960), stream=stream0)
        buf600 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_98.run(buf600, relu_30, cat_17, unsqueeze_1406, buf599, squeeze_91, buf598, primals_61, 6021120, grid=grid(6021120), stream=stream0)
        del buf599
        del cat_17
        del primals_61
        del relu_30
        del squeeze_91
        del unsqueeze_1406
        buf602 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_99.run(buf535, buf556, buf578, buf600, buf602, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf603 = aten.convolution_backward(buf602, relu_29, primals_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_252
        buf604 = buf603[0]
        buf605 = buf603[1]
        del buf603
        buf606 = buf592; del buf592  # reuse
        buf607 = empty((400, ), device='cuda', dtype=torch.float32)
        buf608 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_29, buf604, convolution_28, unsqueeze_1418, squeeze_88, buf606, buf607, buf608, 400, 6272, grid=grid(400), stream=stream0)
        buf609 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf609, relu_29, convolution_28, unsqueeze_1418, buf607, squeeze_88, buf606, primals_59, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_28
        del primals_59
        del relu_29
        del squeeze_88
        del unsqueeze_1418
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf610 = aten.convolution_backward(buf609, relu_28, primals_251, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf609
        del primals_251
        buf611 = buf610[0]
        buf612 = buf610[1]
        del buf610
        buf613 = buf607; del buf607  # reuse
        buf614 = empty((400, ), device='cuda', dtype=torch.float32)
        buf615 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_28, buf611, convolution_27, unsqueeze_1430, squeeze_85, buf613, buf614, buf615, 400, 6272, grid=grid(400), stream=stream0)
        buf616 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf616, relu_28, convolution_27, unsqueeze_1430, buf614, squeeze_85, buf613, primals_57, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_27
        del primals_57
        del relu_28
        del squeeze_85
        del unsqueeze_1430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf617 = aten.convolution_backward(buf616, relu_27, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf616
        del primals_250
        buf618 = buf617[0]
        buf619 = buf617[1]
        del buf617
        buf620 = empty((896, ), device='cuda', dtype=torch.float32)
        buf621 = empty((896, ), device='cuda', dtype=torch.float32)
        buf623 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(relu_27, buf618, cat_15, unsqueeze_1442, squeeze_82, buf620, buf621, buf623, 896, 6272, grid=grid(896), stream=stream0)
        buf622 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_101.run(buf622, relu_27, cat_15, unsqueeze_1442, buf621, squeeze_82, buf620, primals_55, 5619712, grid=grid(5619712), stream=stream0)
        del buf621
        del cat_15
        del primals_55
        del relu_27
        del squeeze_82
        del unsqueeze_1442
        buf624 = reinterpret_tensor(buf213, (8, 512, 28, 28), (401408, 784, 28, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_102.run(buf535, buf556, buf578, buf600, buf622, buf624, 3211264, grid=grid(3211264), stream=stream0)
        buf625 = reinterpret_tensor(buf393, (8, 384, 28, 28), (301056, 784, 28, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_103.run(buf535, buf556, buf578, buf600, buf622, buf625, 2408448, grid=grid(2408448), stream=stream0)
        del buf535
        del buf556
        del buf578
        del buf600
        del buf622
        buf626 = buf602; del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_104.run(buf625, buf624, buf626, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf627 = aten.convolution_backward(buf626, relu_26, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_249
        buf628 = buf627[0]
        buf629 = buf627[1]
        del buf627
        buf630 = buf614; del buf614  # reuse
        buf631 = empty((400, ), device='cuda', dtype=torch.float32)
        buf632 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_26, buf628, convolution_25, unsqueeze_1454, squeeze_79, buf630, buf631, buf632, 400, 6272, grid=grid(400), stream=stream0)
        buf633 = buf628; del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf633, relu_26, convolution_25, unsqueeze_1454, buf631, squeeze_79, buf630, primals_53, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_25
        del primals_53
        del relu_26
        del squeeze_79
        del unsqueeze_1454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf634 = aten.convolution_backward(buf633, relu_25, primals_248, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf633
        del primals_248
        buf635 = buf634[0]
        buf636 = buf634[1]
        del buf634
        buf637 = buf631; del buf631  # reuse
        buf638 = empty((400, ), device='cuda', dtype=torch.float32)
        buf639 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_25, buf635, convolution_24, unsqueeze_1466, squeeze_76, buf637, buf638, buf639, 400, 6272, grid=grid(400), stream=stream0)
        buf640 = buf635; del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf640, relu_25, convolution_24, unsqueeze_1466, buf638, squeeze_76, buf637, primals_51, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_24
        del primals_51
        del relu_25
        del squeeze_76
        del unsqueeze_1466
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf641 = aten.convolution_backward(buf640, relu_24, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf640
        del primals_247
        buf642 = buf641[0]
        buf643 = buf641[1]
        del buf641
        buf644 = empty((832, ), device='cuda', dtype=torch.float32)
        buf645 = empty((832, ), device='cuda', dtype=torch.float32)
        buf647 = empty((832, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_105.run(relu_24, buf642, cat_13, unsqueeze_1478, squeeze_73, buf644, buf645, buf647, 832, 6272, grid=grid(832), stream=stream0)
        buf646 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_106.run(buf646, relu_24, cat_13, unsqueeze_1478, buf645, squeeze_73, buf644, primals_49, 5218304, grid=grid(5218304), stream=stream0)
        del buf645
        del cat_13
        del primals_49
        del relu_24
        del squeeze_73
        del unsqueeze_1478
        buf648 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_107.run(buf625, buf646, buf624, buf648, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf649 = aten.convolution_backward(buf648, relu_23, primals_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_246
        buf650 = buf649[0]
        buf651 = buf649[1]
        del buf649
        buf652 = buf638; del buf638  # reuse
        buf653 = empty((400, ), device='cuda', dtype=torch.float32)
        buf654 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_23, buf650, convolution_22, unsqueeze_1490, squeeze_70, buf652, buf653, buf654, 400, 6272, grid=grid(400), stream=stream0)
        buf655 = buf650; del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf655, relu_23, convolution_22, unsqueeze_1490, buf653, squeeze_70, buf652, primals_47, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_22
        del primals_47
        del relu_23
        del squeeze_70
        del unsqueeze_1490
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf656 = aten.convolution_backward(buf655, relu_22, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf655
        del primals_245
        buf657 = buf656[0]
        buf658 = buf656[1]
        del buf656
        buf659 = buf653; del buf653  # reuse
        buf660 = empty((400, ), device='cuda', dtype=torch.float32)
        buf661 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_22, buf657, convolution_21, unsqueeze_1502, squeeze_67, buf659, buf660, buf661, 400, 6272, grid=grid(400), stream=stream0)
        buf662 = buf657; del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf662, relu_22, convolution_21, unsqueeze_1502, buf660, squeeze_67, buf659, primals_45, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_21
        del primals_45
        del relu_22
        del squeeze_67
        del unsqueeze_1502
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf663 = aten.convolution_backward(buf662, relu_21, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf662
        del primals_244
        buf664 = buf663[0]
        buf665 = buf663[1]
        del buf663
        buf666 = empty((768, ), device='cuda', dtype=torch.float32)
        buf667 = empty((768, ), device='cuda', dtype=torch.float32)
        buf669 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_108.run(relu_21, buf664, cat_11, unsqueeze_1514, squeeze_64, buf666, buf667, buf669, 768, 6272, grid=grid(768), stream=stream0)
        buf668 = buf664; del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_109.run(buf668, relu_21, cat_11, unsqueeze_1514, buf667, squeeze_64, buf666, primals_43, 4816896, grid=grid(4816896), stream=stream0)
        del buf667
        del cat_11
        del primals_43
        del relu_21
        del squeeze_64
        del unsqueeze_1514
        buf670 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_110.run(buf625, buf646, buf668, buf624, buf670, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf671 = aten.convolution_backward(buf670, relu_20, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_243
        buf672 = buf671[0]
        buf673 = buf671[1]
        del buf671
        buf674 = buf660; del buf660  # reuse
        buf675 = empty((400, ), device='cuda', dtype=torch.float32)
        buf676 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_20, buf672, convolution_19, unsqueeze_1526, squeeze_61, buf674, buf675, buf676, 400, 6272, grid=grid(400), stream=stream0)
        buf677 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf677, relu_20, convolution_19, unsqueeze_1526, buf675, squeeze_61, buf674, primals_41, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_19
        del primals_41
        del relu_20
        del squeeze_61
        del unsqueeze_1526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf678 = aten.convolution_backward(buf677, relu_19, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf677
        del primals_242
        buf679 = buf678[0]
        buf680 = buf678[1]
        del buf678
        buf681 = buf675; del buf675  # reuse
        buf682 = empty((400, ), device='cuda', dtype=torch.float32)
        buf683 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_19, buf679, convolution_18, unsqueeze_1538, squeeze_58, buf681, buf682, buf683, 400, 6272, grid=grid(400), stream=stream0)
        buf684 = buf679; del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf684, relu_19, convolution_18, unsqueeze_1538, buf682, squeeze_58, buf681, primals_39, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_18
        del primals_39
        del relu_19
        del squeeze_58
        del unsqueeze_1538
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf685 = aten.convolution_backward(buf684, relu_18, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf684
        del primals_241
        buf686 = buf685[0]
        buf687 = buf685[1]
        del buf685
        buf688 = empty((704, ), device='cuda', dtype=torch.float32)
        buf689 = empty((704, ), device='cuda', dtype=torch.float32)
        buf691 = empty((704, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_111.run(relu_18, buf686, cat_9, unsqueeze_1550, squeeze_55, buf688, buf689, buf691, 704, 6272, grid=grid(704), stream=stream0)
        buf690 = buf686; del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_112.run(buf690, relu_18, cat_9, unsqueeze_1550, buf689, squeeze_55, buf688, primals_37, 4415488, grid=grid(4415488), stream=stream0)
        del buf689
        del cat_9
        del primals_37
        del relu_18
        del squeeze_55
        del unsqueeze_1550
        buf692 = buf670; del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_113.run(buf625, buf646, buf668, buf690, buf624, buf692, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf693 = aten.convolution_backward(buf692, relu_17, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf692
        del primals_240
        buf694 = buf693[0]
        buf695 = buf693[1]
        del buf693
        buf696 = buf682; del buf682  # reuse
        buf697 = empty((400, ), device='cuda', dtype=torch.float32)
        buf698 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_17, buf694, convolution_16, unsqueeze_1562, squeeze_52, buf696, buf697, buf698, 400, 6272, grid=grid(400), stream=stream0)
        buf699 = buf694; del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90.run(buf699, relu_17, convolution_16, unsqueeze_1562, buf697, squeeze_52, buf696, primals_35, 2508800, grid=grid(2508800), stream=stream0)
        del convolution_16
        del primals_35
        del relu_17
        del squeeze_52
        del unsqueeze_1562
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf700 = aten.convolution_backward(buf699, relu_16, primals_239, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf699
        del primals_239
        buf701 = buf700[0]
        buf702 = buf700[1]
        del buf700
        buf703 = buf697; del buf697  # reuse
        buf704 = empty((400, ), device='cuda', dtype=torch.float32)
        buf705 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_114.run(relu_16, buf701, convolution_15, unsqueeze_1574, squeeze_49, buf703, buf704, buf705, 400, 25088, grid=grid(400), stream=stream0)
        buf706 = buf701; del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_115.run(buf706, relu_16, convolution_15, unsqueeze_1574, buf704, squeeze_49, buf703, primals_33, 10035200, grid=grid(10035200), stream=stream0)
        del buf704
        del convolution_15
        del primals_33
        del relu_16
        del squeeze_49
        del unsqueeze_1574
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf707 = aten.convolution_backward(buf706, relu_15, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf706
        del primals_238
        buf708 = buf707[0]
        buf709 = buf707[1]
        del buf707
        buf713 = empty((8, 640, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_116.run(buf625, buf646, buf668, buf690, buf624, buf713, 4014080, grid=grid(4014080), stream=stream0)
        del buf624
        del buf625
        del buf646
        del buf668
        del buf690
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf714 = aten.convolution_backward(buf713, relu_14, primals_237, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf713
        del primals_237
        buf715 = buf714[0]
        buf710 = empty((376, ), device='cuda', dtype=torch.float32)
        buf711 = empty((376, ), device='cuda', dtype=torch.float32)
        buf717 = empty((376, ), device='cuda', dtype=torch.float32)
        buf718 = empty((376, ), device='cuda', dtype=torch.float32)
        buf712 = empty((376, ), device='cuda', dtype=torch.float32)
        buf719 = empty((376, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_117.run(relu_15, buf708, cat_7, unsqueeze_1586, relu_14, buf715, squeeze_43, buf710, buf711, buf717, buf718, buf712, buf719, 376, 25088, grid=grid(376), stream=stream0)
        buf716 = buf714[1]
        del buf714
        buf720 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_118.run(buf720, relu_15, cat_7, unsqueeze_1586, buf711, squeeze_43, buf710, primals_31, relu_14, buf715, buf718, buf717, primals_29, 9433088, grid=grid(9433088), stream=stream0)
        del buf711
        del buf715
        del buf718
        del cat_7
        del primals_29
        del primals_31
        del relu_14
        del relu_15
        del squeeze_43
        del unsqueeze_1586
        buf721 = empty((8, 276, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_119.run(buf720, buf721, 6924288, grid=grid(6924288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf722 = aten.convolution_backward(buf721, relu_13, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_236
        buf723 = buf722[0]
        buf724 = buf722[1]
        del buf722
        buf725 = reinterpret_tensor(buf519, (200, 4), (1, 200), 0); del buf519  # reuse
        buf727 = empty_strided((200, 4), (1, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_13, buf723, convolution_12, unsqueeze_1610, buf725, buf727, 800, 6272, grid=grid(800), stream=stream0)
        buf726 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf725, buf726, 200, 4, grid=grid(200), stream=stream0)
        buf728 = empty((200, ), device='cuda', dtype=torch.float32)
        buf729 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf727, squeeze_40, buf728, buf729, 200, 4, grid=grid(200), stream=stream0)
        buf730 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf730, relu_13, convolution_12, unsqueeze_1610, buf728, squeeze_40, buf726, primals_27, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_12
        del primals_27
        del relu_13
        del squeeze_40
        del unsqueeze_1610
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf731 = aten.convolution_backward(buf730, relu_12, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf730
        del primals_235
        buf732 = buf731[0]
        buf733 = buf731[1]
        del buf731
        buf734 = buf727; del buf727  # reuse
        buf736 = buf725; del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_12, buf732, convolution_11, unsqueeze_1622, buf734, buf736, 800, 6272, grid=grid(800), stream=stream0)
        buf735 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf734, buf735, 200, 4, grid=grid(200), stream=stream0)
        buf737 = empty((200, ), device='cuda', dtype=torch.float32)
        buf738 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf736, squeeze_37, buf737, buf738, 200, 4, grid=grid(200), stream=stream0)
        buf739 = buf732; del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf739, relu_12, convolution_11, unsqueeze_1622, buf737, squeeze_37, buf735, primals_25, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_11
        del primals_25
        del relu_12
        del squeeze_37
        del unsqueeze_1622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf740 = aten.convolution_backward(buf739, relu_11, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf739
        del primals_234
        buf741 = buf740[0]
        buf742 = buf740[1]
        del buf740
        buf743 = empty((356, ), device='cuda', dtype=torch.float32)
        buf744 = empty((356, ), device='cuda', dtype=torch.float32)
        buf746 = empty((356, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_124.run(relu_11, buf741, cat_5, unsqueeze_1634, squeeze_34, buf743, buf744, buf746, 356, 25088, grid=grid(356), stream=stream0)
        buf745 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_125.run(buf745, relu_11, cat_5, unsqueeze_1634, buf744, squeeze_34, buf743, primals_23, 8931328, grid=grid(8931328), stream=stream0)
        del buf744
        del cat_5
        del primals_23
        del relu_11
        del squeeze_34
        del unsqueeze_1634
        buf747 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_126.run(buf720, buf745, buf747, 6924288, grid=grid(6924288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf748 = aten.convolution_backward(buf747, relu_10, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_233
        buf749 = buf748[0]
        buf750 = buf748[1]
        del buf748
        buf751 = buf736; del buf736  # reuse
        buf753 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_10, buf749, convolution_9, unsqueeze_1646, buf751, buf753, 800, 6272, grid=grid(800), stream=stream0)
        buf752 = buf737; del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf751, buf752, 200, 4, grid=grid(200), stream=stream0)
        buf754 = empty((200, ), device='cuda', dtype=torch.float32)
        buf755 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf753, squeeze_31, buf754, buf755, 200, 4, grid=grid(200), stream=stream0)
        buf756 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf756, relu_10, convolution_9, unsqueeze_1646, buf754, squeeze_31, buf752, primals_21, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_9
        del primals_21
        del relu_10
        del squeeze_31
        del unsqueeze_1646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf757 = aten.convolution_backward(buf756, relu_9, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf756
        del primals_232
        buf758 = buf757[0]
        buf759 = buf757[1]
        del buf757
        buf760 = buf753; del buf753  # reuse
        buf762 = buf751; del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_9, buf758, convolution_8, unsqueeze_1658, buf760, buf762, 800, 6272, grid=grid(800), stream=stream0)
        buf761 = buf754; del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf760, buf761, 200, 4, grid=grid(200), stream=stream0)
        buf763 = empty((200, ), device='cuda', dtype=torch.float32)
        buf764 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf762, squeeze_28, buf763, buf764, 200, 4, grid=grid(200), stream=stream0)
        buf765 = buf758; del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf765, relu_9, convolution_8, unsqueeze_1658, buf763, squeeze_28, buf761, primals_19, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_8
        del primals_19
        del relu_9
        del squeeze_28
        del unsqueeze_1658
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf766 = aten.convolution_backward(buf765, relu_8, primals_231, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf765
        del primals_231
        buf767 = buf766[0]
        buf768 = buf766[1]
        del buf766
        buf769 = empty((336, ), device='cuda', dtype=torch.float32)
        buf770 = empty((336, ), device='cuda', dtype=torch.float32)
        buf772 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_127.run(relu_8, buf767, cat_3, unsqueeze_1670, squeeze_25, buf769, buf770, buf772, 336, 25088, grid=grid(336), stream=stream0)
        buf771 = buf767; del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_128.run(buf771, relu_8, cat_3, unsqueeze_1670, buf770, squeeze_25, buf769, primals_17, 8429568, grid=grid(8429568), stream=stream0)
        del buf770
        del cat_3
        del primals_17
        del relu_8
        del squeeze_25
        del unsqueeze_1670
        buf773 = buf747; del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_129.run(buf720, buf745, buf771, buf773, 6924288, grid=grid(6924288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf774 = aten.convolution_backward(buf773, relu_7, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_230
        buf775 = buf774[0]
        buf776 = buf774[1]
        del buf774
        buf777 = buf762; del buf762  # reuse
        buf779 = buf760; del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_7, buf775, convolution_6, unsqueeze_1682, buf777, buf779, 800, 6272, grid=grid(800), stream=stream0)
        buf778 = buf763; del buf763  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf777, buf778, 200, 4, grid=grid(200), stream=stream0)
        buf780 = empty((200, ), device='cuda', dtype=torch.float32)
        buf781 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf779, squeeze_22, buf780, buf781, 200, 4, grid=grid(200), stream=stream0)
        buf782 = buf775; del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf782, relu_7, convolution_6, unsqueeze_1682, buf780, squeeze_22, buf778, primals_15, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_6
        del primals_15
        del relu_7
        del squeeze_22
        del unsqueeze_1682
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf783 = aten.convolution_backward(buf782, relu_6, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf782
        del primals_229
        buf784 = buf783[0]
        buf785 = buf783[1]
        del buf783
        buf786 = buf779; del buf779  # reuse
        buf788 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_6, buf784, convolution_5, unsqueeze_1694, buf786, buf788, 800, 6272, grid=grid(800), stream=stream0)
        buf787 = buf780; del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf786, buf787, 200, 4, grid=grid(200), stream=stream0)
        buf789 = empty((200, ), device='cuda', dtype=torch.float32)
        buf790 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf788, squeeze_19, buf789, buf790, 200, 4, grid=grid(200), stream=stream0)
        buf791 = buf784; del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf791, relu_6, convolution_5, unsqueeze_1694, buf789, squeeze_19, buf787, primals_13, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_5
        del primals_13
        del relu_6
        del squeeze_19
        del unsqueeze_1694
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf792 = aten.convolution_backward(buf791, relu_5, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf791
        del primals_228
        buf793 = buf792[0]
        buf794 = buf792[1]
        del buf792
        buf795 = empty((316, ), device='cuda', dtype=torch.float32)
        buf796 = empty((316, ), device='cuda', dtype=torch.float32)
        buf798 = empty((316, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_130.run(relu_5, buf793, cat_1, unsqueeze_1706, squeeze_16, buf795, buf796, buf798, 316, 25088, grid=grid(316), stream=stream0)
        buf797 = buf793; del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_131.run(buf797, relu_5, cat_1, unsqueeze_1706, buf796, squeeze_16, buf795, primals_11, 7927808, grid=grid(7927808), stream=stream0)
        del buf796
        del cat_1
        del primals_11
        del relu_5
        del squeeze_16
        del unsqueeze_1706
        buf799 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_132.run(buf720, buf745, buf771, buf797, buf799, 6924288, grid=grid(6924288), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf800 = aten.convolution_backward(buf799, relu_4, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf799
        del primals_227
        buf801 = buf800[0]
        buf802 = buf800[1]
        del buf800
        buf803 = buf788; del buf788  # reuse
        buf805 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_4, buf801, convolution_3, unsqueeze_1718, buf803, buf805, 800, 6272, grid=grid(800), stream=stream0)
        buf804 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf803, buf804, 200, 4, grid=grid(200), stream=stream0)
        buf806 = empty((200, ), device='cuda', dtype=torch.float32)
        buf807 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf805, squeeze_13, buf806, buf807, 200, 4, grid=grid(200), stream=stream0)
        buf808 = buf801; del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf808, relu_4, convolution_3, unsqueeze_1718, buf806, squeeze_13, buf804, primals_9, 5017600, grid=grid(5017600), stream=stream0)
        del convolution_3
        del primals_9
        del relu_4
        del squeeze_13
        del unsqueeze_1718
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf809 = aten.convolution_backward(buf808, relu_3, primals_226, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
        del buf808
        del primals_226
        buf810 = buf809[0]
        buf811 = buf809[1]
        del buf809
        buf812 = buf805; del buf805  # reuse
        buf814 = buf803; del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(relu_3, buf810, convolution_2, unsqueeze_1730, buf812, buf814, 800, 6272, grid=grid(800), stream=stream0)
        buf813 = buf806; del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf812, buf813, 200, 4, grid=grid(200), stream=stream0)
        del buf812
        buf815 = empty((200, ), device='cuda', dtype=torch.float32)
        buf816 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_122.run(buf814, squeeze_10, buf815, buf816, 200, 4, grid=grid(200), stream=stream0)
        del buf814
        buf817 = buf810; del buf810  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_123.run(buf817, relu_3, convolution_2, unsqueeze_1730, buf815, squeeze_10, buf813, primals_7, 5017600, grid=grid(5017600), stream=stream0)
        del buf815
        del convolution_2
        del primals_7
        del relu_3
        del squeeze_10
        del unsqueeze_1730
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf818 = aten.convolution_backward(buf817, relu_2, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf817
        del primals_225
        buf819 = buf818[0]
        buf820 = buf818[1]
        del buf818
        buf826 = empty((8, 296, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_slice_backward_133.run(buf720, buf745, buf771, buf797, buf826, 7426048, grid=grid(7426048), stream=stream0)
        del buf720
        del buf745
        del buf771
        del buf797
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
        buf827 = aten.convolution_backward(buf826, relu_1, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf826
        del primals_224
        buf828 = buf827[0]
        buf821 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf823 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf830 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf832 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_134.run(relu_2, buf819, sub_543, relu_1, buf828, buf821, buf823, buf830, buf832, 512, 6272, grid=grid(512), stream=stream0)
        buf822 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_135.run(buf821, buf822, 128, 4, grid=grid(128), stream=stream0)
        del buf821
        buf833 = empty((128, ), device='cuda', dtype=torch.float32)
        buf824 = empty((128, ), device='cuda', dtype=torch.float32)
        buf825 = empty((128, ), device='cuda', dtype=torch.float32)
        buf834 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_136.run(buf832, buf823, squeeze_4, buf833, buf824, buf825, buf834, 128, 4, grid=grid(128), stream=stream0)
        del buf823
        buf829 = buf827[1]
        del buf827
        buf831 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_135.run(buf830, buf831, 128, 4, grid=grid(128), stream=stream0)
        buf835 = buf819; del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_137.run(buf835, relu_2, sub_543, buf824, squeeze_4, buf822, primals_5, relu_1, buf828, buf833, buf831, primals_3, 3211264, grid=grid(3211264), stream=stream0)
        del buf828
        del primals_3
        del primals_5
        del relu_1
        del relu_2
        del squeeze_4
        del sub_543
        buf836 = empty((8, 128, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        triton_poi_fused_max_pool2d_with_indices_backward_138.run(getitem_3, buf835, buf836, 12845056, grid=grid(12845056), stream=stream0)
        del buf835
        del getitem_3
        buf837 = buf830; del buf830  # reuse
        buf839 = buf832; del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_139.run(relu, buf836, convolution, unsqueeze_1766, buf837, buf839, 512, 25088, grid=grid(512), stream=stream0)
        buf838 = buf833; del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_135.run(buf837, buf838, 128, 4, grid=grid(128), stream=stream0)
        del buf837
        buf840 = buf824; del buf824  # reuse
        buf841 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_140.run(buf839, squeeze_1, buf840, buf841, 128, 4, grid=grid(128), stream=stream0)
        del buf839
        buf842 = buf836; del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_141.run(buf842, relu, convolution, unsqueeze_1766, buf840, squeeze_1, buf838, primals_1, 12845056, grid=grid(12845056), stream=stream0)
        del buf840
        del convolution
        del primals_1
        del relu
        del squeeze_1
        del unsqueeze_1766
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf843 = aten.convolution_backward(buf842, primals_668, primals_223, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf842
        del primals_223
        del primals_668
        buf844 = buf843[1]
        return (buf841, buf838, buf834, buf831, buf825, buf822, buf816, buf813, buf807, buf804, buf798, buf795, buf790, buf787, buf781, buf778, buf772, buf769, buf764, buf761, buf755, buf752, buf746, buf743, buf738, buf735, buf729, buf726, buf719, buf717, buf712, buf710, buf705, buf703, buf698, buf696, buf691, buf688, buf683, buf681, buf676, buf674, buf669, buf666, buf661, buf659, buf654, buf652, buf647, buf644, buf639, buf637, buf632, buf630, buf623, buf620, buf615, buf613, buf608, buf606, buf601, buf598, buf593, buf591, buf586, buf584, buf579, buf576, buf571, buf569, buf564, buf562, buf557, buf554, buf549, buf547, buf542, buf540, buf534, buf532, buf527, buf525, buf520, buf518, buf513, buf511, buf506, buf503, buf498, buf496, buf491, buf489, buf484, buf481, buf476, buf474, buf469, buf467, buf462, buf459, buf454, buf452, buf447, buf445, buf438, buf435, buf430, buf428, buf423, buf421, buf416, buf413, buf408, buf406, buf401, buf399, buf394, buf391, buf386, buf384, buf379, buf377, buf372, buf369, buf364, buf362, buf357, buf355, buf348, buf345, buf340, buf338, buf333, buf331, buf326, buf323, buf318, buf316, buf311, buf309, buf304, buf301, buf296, buf294, buf289, buf287, buf282, buf279, buf274, buf272, buf267, buf265, buf258, buf255, buf250, buf248, buf243, buf241, buf236, buf233, buf228, buf226, buf221, buf219, buf214, buf211, buf206, buf204, buf199, buf197, buf192, buf189, buf184, buf182, buf177, buf175, buf168, buf165, buf160, buf158, buf153, buf151, buf146, buf143, buf138, buf136, buf131, buf129, buf124, buf121, buf116, buf114, buf109, buf107, buf102, buf99, buf94, buf92, buf87, buf85, buf79, buf77, buf72, buf70, buf65, buf63, buf58, buf56, buf51, buf48, buf43, buf41, buf36, buf34, buf29, buf26, buf21, buf19, buf14, buf12, buf7, buf4, buf844, buf829, buf820, buf811, buf802, buf794, buf785, buf776, buf768, buf759, buf750, buf742, buf733, buf724, buf716, buf709, buf702, buf695, buf687, buf680, buf673, buf665, buf658, buf651, buf643, buf636, buf629, buf619, buf612, buf605, buf597, buf590, buf583, buf575, buf568, buf561, buf553, buf546, buf539, buf531, buf524, buf517, buf510, buf502, buf495, buf488, buf480, buf473, buf466, buf458, buf451, buf444, buf434, buf427, buf420, buf412, buf405, buf398, buf390, buf383, buf376, buf368, buf361, buf354, buf344, buf337, buf330, buf322, buf315, buf308, buf300, buf293, buf286, buf278, buf271, buf264, buf254, buf247, buf240, buf232, buf225, buf218, buf210, buf203, buf196, buf188, buf181, buf174, buf164, buf157, buf150, buf142, buf135, buf128, buf120, buf113, buf106, buf98, buf91, buf84, buf76, buf69, buf62, buf55, buf47, buf40, buf33, buf25, buf18, buf11, buf3, buf0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((296, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((200, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((200, 316, 1, 1), (316, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((200, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((200, 356, 1, 1), (356, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((640, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((400, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((400, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((400, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((400, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((400, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((400, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((400, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((400, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1152, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((800, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((800, 1216, 1, 1), (1216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((800, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((800, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((800, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((800, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((800, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((800, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((800, 1664, 1, 1), (1664, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((800, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((800, 1792, 1, 1), (1792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((800, 1856, 1, 1), (1856, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((800, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((800, 1984, 1, 1), (1984, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((800, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((800, 2112, 1, 1), (2112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((800, 2176, 1, 1), (2176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((800, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((800, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((800, 2368, 1, 1), (2368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((2304, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1600, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1000, 2688, 1, 1), (2688, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 128, 112, 112), (1605632, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 128, 112, 112), (1605632, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.int64)
    squeeze_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 316, 56, 56), (990976, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 316, 56, 56), (990976, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 336, 56, 56), (1053696, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 336, 56, 56), (1053696, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 356, 56, 56), (1116416, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 356, 56, 56), (1116416, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 200, 56, 56), (627200, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 376, 56, 56), (1179136, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 376, 56, 56), (1179136, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 376, 56, 56), (1179136, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 400, 56, 56), (1254400, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 400, 56, 56), (1254400, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 704, 28, 28), (551936, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 704, 28, 28), (551936, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 768, 28, 28), (602112, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 768, 28, 28), (602112, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_13 = rand_strided((8, 832, 28, 28), (652288, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 832, 28, 28), (652288, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_15 = rand_strided((8, 896, 28, 28), (702464, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 896, 28, 28), (702464, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_17 = rand_strided((8, 960, 28, 28), (752640, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 960, 28, 28), (752640, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_19 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_21 = rand_strided((8, 1088, 28, 28), (852992, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 1088, 28, 28), (852992, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 400, 28, 28), (313600, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_23 = rand_strided((8, 1152, 28, 28), (903168, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 1152, 28, 28), (903168, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 1152, 28, 28), (903168, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 800, 28, 28), (627200, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 800, 28, 28), (627200, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_25 = rand_strided((8, 1216, 14, 14), (238336, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((8, 1216, 14, 14), (238336, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_27 = rand_strided((8, 1280, 14, 14), (250880, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 1280, 14, 14), (250880, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_29 = rand_strided((8, 1344, 14, 14), (263424, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((8, 1344, 14, 14), (263424, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_31 = rand_strided((8, 1408, 14, 14), (275968, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_52 = rand_strided((8, 1408, 14, 14), (275968, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_53 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_54 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_33 = rand_strided((8, 1472, 14, 14), (288512, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((8, 1472, 14, 14), (288512, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_56 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_57 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_35 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_58 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_59 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_37 = rand_strided((8, 1600, 14, 14), (313600, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_61 = rand_strided((8, 1600, 14, 14), (313600, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_62 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_63 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_39 = rand_strided((8, 1664, 14, 14), (326144, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_64 = rand_strided((8, 1664, 14, 14), (326144, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_66 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_41 = rand_strided((8, 1728, 14, 14), (338688, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_67 = rand_strided((8, 1728, 14, 14), (338688, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_68 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_69 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_43 = rand_strided((8, 1792, 14, 14), (351232, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((8, 1792, 14, 14), (351232, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_71 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_72 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_45 = rand_strided((8, 1856, 14, 14), (363776, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_73 = rand_strided((8, 1856, 14, 14), (363776, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_74 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_47 = rand_strided((8, 1920, 14, 14), (376320, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_76 = rand_strided((8, 1920, 14, 14), (376320, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_77 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_78 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_49 = rand_strided((8, 1984, 14, 14), (388864, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_79 = rand_strided((8, 1984, 14, 14), (388864, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_80 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_81 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_51 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_82 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_83 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_84 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_53 = rand_strided((8, 2112, 14, 14), (413952, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_85 = rand_strided((8, 2112, 14, 14), (413952, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_86 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_262 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_87 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_55 = rand_strided((8, 2176, 14, 14), (426496, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_265 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_88 = rand_strided((8, 2176, 14, 14), (426496, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_268 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_89 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_271 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_90 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_57 = rand_strided((8, 2240, 14, 14), (439040, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_274 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_91 = rand_strided((8, 2240, 14, 14), (439040, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_277 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_92 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_280 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_93 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_59 = rand_strided((8, 2304, 14, 14), (451584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_283 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_94 = rand_strided((8, 2304, 14, 14), (451584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_286 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_95 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_289 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_96 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_61 = rand_strided((8, 2368, 14, 14), (464128, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_292 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_97 = rand_strided((8, 2368, 14, 14), (464128, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_97 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_295 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_98 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_98 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_298 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_99 = rand_strided((8, 800, 14, 14), (156800, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_63 = rand_strided((8, 2432, 14, 14), (476672, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_301 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_100 = rand_strided((8, 2432, 14, 14), (476672, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_101 = rand_strided((8, 2432, 14, 14), (476672, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_101 = rand_strided((8, 1600, 14, 14), (313600, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_307 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_102 = rand_strided((8, 1600, 14, 14), (313600, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_102 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_310 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_103 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_65 = rand_strided((8, 2432, 7, 7), (119168, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_313 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_104 = rand_strided((8, 2432, 7, 7), (119168, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_104 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_316 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_105 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_105 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_319 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_106 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_67 = rand_strided((8, 2560, 7, 7), (125440, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_322 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_107 = rand_strided((8, 2560, 7, 7), (125440, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_107 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_325 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_108 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_108 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_328 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_109 = rand_strided((8, 1600, 7, 7), (78400, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_69 = rand_strided((8, 2688, 7, 7), (131712, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_331 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 2688, 1, 1), (2688, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 2688, 7, 7), (131712, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_446 = rand_strided((1, 2688, 1, 1), (2688, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_518 = rand_strided((1, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_530 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_554 = rand_strided((1, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_590 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 2368, 1, 1), (2368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_614 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_626 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_662 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_698 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 2176, 1, 1), (2176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_734 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 2112, 1, 1), (2112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_782 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_806 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 1984, 1, 1), (1984, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_866 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 1856, 1, 1), (1856, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_914 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_926 = rand_strided((1, 1792, 1, 1), (1792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_938 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_950 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_962 = rand_strided((1, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_974 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_986 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_998 = rand_strided((1, 1664, 1, 1), (1664, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1010 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1022 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1034 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1046 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1058 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1070 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1082 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1094 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1106 = rand_strided((1, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1118 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1130 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1142 = rand_strided((1, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1154 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1166 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1178 = rand_strided((1, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1190 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1202 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1214 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1226 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1238 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1250 = rand_strided((1, 1216, 1, 1), (1216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1262 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1274 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1286 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1310 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1322 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1334 = rand_strided((1, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1346 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1358 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1370 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1382 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1394 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1406 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1418 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1430 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1442 = rand_strided((1, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1454 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1466 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1478 = rand_strided((1, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1490 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1502 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1514 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1526 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1538 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1550 = rand_strided((1, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1562 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1574 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1586 = rand_strided((1, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1610 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1622 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1634 = rand_strided((1, 356, 1, 1), (356, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1646 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1658 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1670 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1682 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1694 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1706 = rand_strided((1, 316, 1, 1), (316, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1718 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1730 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_543 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1766 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_668, convolution, squeeze_1, relu, getitem_3, squeeze_4, relu_1, relu_2, convolution_2, squeeze_10, relu_3, convolution_3, squeeze_13, relu_4, cat_1, squeeze_16, relu_5, convolution_5, squeeze_19, relu_6, convolution_6, squeeze_22, relu_7, cat_3, squeeze_25, relu_8, convolution_8, squeeze_28, relu_9, convolution_9, squeeze_31, relu_10, cat_5, squeeze_34, relu_11, convolution_11, squeeze_37, relu_12, convolution_12, squeeze_40, relu_13, cat_7, squeeze_43, relu_14, relu_15, convolution_15, squeeze_49, relu_16, convolution_16, squeeze_52, relu_17, cat_9, squeeze_55, relu_18, convolution_18, squeeze_58, relu_19, convolution_19, squeeze_61, relu_20, cat_11, squeeze_64, relu_21, convolution_21, squeeze_67, relu_22, convolution_22, squeeze_70, relu_23, cat_13, squeeze_73, relu_24, convolution_24, squeeze_76, relu_25, convolution_25, squeeze_79, relu_26, cat_15, squeeze_82, relu_27, convolution_27, squeeze_85, relu_28, convolution_28, squeeze_88, relu_29, cat_17, squeeze_91, relu_30, convolution_30, squeeze_94, relu_31, convolution_31, squeeze_97, relu_32, cat_19, squeeze_100, relu_33, convolution_33, squeeze_103, relu_34, convolution_34, squeeze_106, relu_35, cat_21, squeeze_109, relu_36, convolution_36, squeeze_112, relu_37, convolution_37, squeeze_115, relu_38, cat_23, squeeze_118, relu_39, relu_40, convolution_40, squeeze_124, relu_41, convolution_41, squeeze_127, relu_42, cat_25, squeeze_130, relu_43, convolution_43, squeeze_133, relu_44, convolution_44, squeeze_136, relu_45, cat_27, squeeze_139, relu_46, convolution_46, squeeze_142, relu_47, convolution_47, squeeze_145, relu_48, cat_29, squeeze_148, relu_49, convolution_49, squeeze_151, relu_50, convolution_50, squeeze_154, relu_51, cat_31, squeeze_157, relu_52, convolution_52, squeeze_160, relu_53, convolution_53, squeeze_163, relu_54, cat_33, squeeze_166, relu_55, convolution_55, squeeze_169, relu_56, convolution_56, squeeze_172, relu_57, cat_35, squeeze_175, relu_58, convolution_58, squeeze_178, relu_59, convolution_59, squeeze_181, relu_60, cat_37, squeeze_184, relu_61, convolution_61, squeeze_187, relu_62, convolution_62, squeeze_190, relu_63, cat_39, squeeze_193, relu_64, convolution_64, squeeze_196, relu_65, convolution_65, squeeze_199, relu_66, cat_41, squeeze_202, relu_67, convolution_67, squeeze_205, relu_68, convolution_68, squeeze_208, relu_69, cat_43, squeeze_211, relu_70, convolution_70, squeeze_214, relu_71, convolution_71, squeeze_217, relu_72, cat_45, squeeze_220, relu_73, convolution_73, squeeze_223, relu_74, convolution_74, squeeze_226, relu_75, cat_47, squeeze_229, relu_76, convolution_76, squeeze_232, relu_77, convolution_77, squeeze_235, relu_78, cat_49, squeeze_238, relu_79, convolution_79, squeeze_241, relu_80, convolution_80, squeeze_244, relu_81, cat_51, squeeze_247, relu_82, convolution_82, squeeze_250, relu_83, convolution_83, squeeze_253, relu_84, cat_53, squeeze_256, relu_85, convolution_85, squeeze_259, relu_86, convolution_86, squeeze_262, relu_87, cat_55, squeeze_265, relu_88, convolution_88, squeeze_268, relu_89, convolution_89, squeeze_271, relu_90, cat_57, squeeze_274, relu_91, convolution_91, squeeze_277, relu_92, convolution_92, squeeze_280, relu_93, cat_59, squeeze_283, relu_94, convolution_94, squeeze_286, relu_95, convolution_95, squeeze_289, relu_96, cat_61, squeeze_292, relu_97, convolution_97, squeeze_295, relu_98, convolution_98, squeeze_298, relu_99, cat_63, squeeze_301, relu_100, relu_101, convolution_101, squeeze_307, relu_102, convolution_102, squeeze_310, relu_103, cat_65, squeeze_313, relu_104, convolution_104, squeeze_316, relu_105, convolution_105, squeeze_319, relu_106, cat_67, squeeze_322, relu_107, convolution_107, squeeze_325, relu_108, convolution_108, squeeze_328, relu_109, cat_69, squeeze_331, mean, le, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1682, unsqueeze_1694, unsqueeze_1706, unsqueeze_1718, unsqueeze_1730, sub_543, unsqueeze_1766, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dpn107', benchmark_compiled_module)
