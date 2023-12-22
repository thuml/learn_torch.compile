
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg463u26szk2koomsmwdhziq76kysbb7oomrurjsyrliqijjwz4.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1000 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2000 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3000 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/f6/cf63s37kdpcwsbp3badzmmvscvxn6kdzqydxv7l7bkccoa7xdz4q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygpwvrnzubejon2vlml4z6g6aurbmv3fsstu6kmmipwdfyempsd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqh3iu33c55xyzvj4m4kr6h7rfzx2az4mnvrvxzfbwljraknjeb.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr6 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp18 <= tmp4
    tmp21 = tmp5 + tmp20
    tmp22 = tl.where(tmp19, tmp4, tmp21)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp29 = tmp27 - tmp28
    tmp30 = tmp22 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp17 * tmp38
    tmp41 = tmp40 + tmp36
    tmp42 = tl.math.rsqrt(tmp41)
    tmp43 = tmp34 * tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp39, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp43, None)
    tl.store(out_ptr0 + (x0), tmp9, None)
    tl.store(out_ptr1 + (x0), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbxh3z6xfixpazw53ypeyiq7dxczq4gr6ttzigbqbvmqprqlnda.py
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
    size_hints=[1024, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c223pbvhc6h7te74jiqae35i76piqcj2kcqtz2gc2hftdbnjiixs.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x3), None)
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pgrvpaf65kgp6zw6zgfh6hpbgfc6zhfa4nfjnald7o4kor3eha.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacgoayoydio5bm3pl5puidqepixs76snj6tmimnl6ewi75ypfp7.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    x3 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None).to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr0 + (x2), None)
    tmp13 = tl.load(in_ptr4 + (x2), None)
    tmp16 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 * tmp21
    tmp24 = tmp23 + tmp17
    tmp25 = tl.math.rsqrt(tmp24)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tl.store(in_out_ptr0 + (x2), tmp15, None)
    tl.store(out_ptr0 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkhfzjc56m52cwhwt75o5pbxkkqz2k4ytmdbkbtx4ocrwejm73w.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = tmp13 - tmp14
    tmp16 = tmp0 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp12 * tmp24
    tmp27 = tmp26 + tmp22
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp20 * tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, None)
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pxxmnofb5cobdc73pn7g726o6mcgz5evnlpyqhhz7qidutbmyl.py
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
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlf2qtrqsyx3y472k32tihuizqc76whne3cpsspgn5lhpe2ersz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clptv6q5jdreotf7kea2mon4pboji5ijzn6lvquvkzhmspadgxye.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wlh2yidy5qm6rd5v6bf2bzihczpulsl6tvrduxzldj3pyrnbee.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogh4elzlj32h5gyvv22qsalkl7oytdvrc4fwrueud33mzt2kd7q.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxei5mgo2uevzrkzjz3cl2jyly5dfzrhpwjvnjj52gbzsdqrfxqz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmazg7gleqgfxpixbzfx4xk6tnftyx45wp4wogyp34ubrnfbxyx.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvt7gbomvqcqshxizjn2yydnkyy6etux7rjmopvt7tdrjwnhtbc4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gks3xlgqeo4sc4pnnb2vjin22peouyzgkurpglqujdd4yi7xz7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 0.0
    tmp15 = tmp13 <= tmp14
    tmp17 = tmp0 + tmp16
    tmp18 = tl.where(tmp15, tmp14, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp25 = tmp23 - tmp24
    tmp26 = tmp18 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp12 * tmp34
    tmp37 = tmp36 + tmp32
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp30 * tmp38
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp35, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp39, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqw3xyewtws3ebseuyyc7vdva6v5eyi37lexgydg4fzv47eto57.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp6 = tl.load(in_out_ptr0 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttumijcu2m5hevm5dwu4yhkrfeg3amt7omzp554a6wtk3cseny3.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19 + tmp13
    tmp21 = tl.math.rsqrt(tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp11 * tmp23
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnab56ode7jdo3xdyzzrpspgifwpfbejkqpwjxslg3dr2ywpnsd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 - tmp14
    tmp16 = tmp0 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp12 * tmp24
    tmp27 = tmp26 + tmp22
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp20 * tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfajjbmpprlsrzn45zdhnzjo5oeggsjekljv7hupae72vqexexnj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp19 = tmp13 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cduu6dlh6kmzfioq6343nbe2jb5zcobk7spxusf3gdgbhbyjsjce.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbj5mjnq2z27zqcs5rpzd2h7ftl2gveio5f2uyyrdqavbf66ivy2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp15 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4s/c4ssd5pqpc3rn2jretzptkoo7vtulrvzl2qfntft2zuimoe6dqsv.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5o5gdgias5milnlxxwt6zznjlary3pvvvmjztypuvnanpfad6j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 3136
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp19 = tmp13 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jba74r7enhiiiebbyp73y2on7tmw6e2bac7lyu5zwouipnxy2s.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6pkfqom3iwdqtpmt6d6hvfhy3qp5f6uj3hr6lflkd73j2nblqt.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxeuaynx7jitgkjergib4eo4gbhvc2dksnk5exjhr5yeozjko5tt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hgyyrjjzmy7dks5tmnzycx3ukvjyd3bmorsqi6rvavbclimkpc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = 0.0
        tmp13 = tmp11 <= tmp12
        tmp15 = tmp0 + tmp14
        tmp16 = tl.where(tmp13, tmp12, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp27 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp9 * tmp30
    tmp33 = tmp32 + tmp28
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp25 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23wtnaz534mwowjwlny4isu2ilxaufy22thbddidj63ep2bemzi.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19 + tmp13
    tmp21 = tl.math.rsqrt(tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp11 * tmp23
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6urigcwqun6ulga6z7ft7ge4hjxlao3rkm6qfsafy5gvxrkzt6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp9 * tmp21
    tmp24 = tmp23 + tmp19
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp16 * tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vaom6xb6fbejostryy565hktvhjp2nqz5ufxbwmzgyfokuqob4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 12544
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
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp19 = tmp13 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xfframvlm4zy4ngwcief6sowuvouzzqaxx6ghc27iseh4l4cvj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sd/csdspgcze5jfof4se4n5hkcntyxppvr555wwizdjgqwrsvdeb6nv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp15 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbeny56q57iye3akxtbcy2emhwpksfzzj6zac6kkdt5uqo5uf6xy.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfd6rjdfn3vgsmfkzfnsz5gbfmpdisjch6jkt5hyjwwzhpevggjg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/we/cwekotvyqvgfu2hmeduczvbmcptdglgui6bphctby3dhaa4m2wek.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/pi/cpi5wftw2w5efp3lolz4q6om3t4yc3vcxeucztjvvuja3tyfvsjw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7bzjkixrgjee3hfrkgpoaqregmhgmmw2ktmwgaya4blcfyspz3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ns/cnsav6wblqer7xa2yj25bbw5x6bhbkv2sxiayimyi5x7rwey7c3i.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(in_out_ptr0 + (x0), tmp11, None)
    tl.store(out_ptr0 + (x0), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokiettvj7uzzqz2lhyt4szrzqddpzipdrjlyqhf5dtbnic6jgwt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t6/ct6we577gdm6roxop2jay4farmrddsf66bqi6saglupoqhmcb2kz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp28 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr7 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = 0.0
        tmp13 = tmp11 <= tmp12
        tmp15 = tmp0 + tmp14
        tmp16 = tl.where(tmp13, tmp12, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tmp29 = tmp27 - tmp28
        tmp30 = tmp16 * tmp29
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tmp34 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp9 * tmp37
    tmp40 = tmp39 + tmp35
    tmp41 = tl.math.rsqrt(tmp40)
    tmp42 = tmp25 * tmp41
    tmp44 = tmp43 + tmp35
    tmp45 = tl.math.rsqrt(tmp44)
    tmp46 = tmp32 * tmp45
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp38, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp42, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp46, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zj/czjlx23oryqeeyiwi7rqodsmnrftbo2pkjbvk66fu2q6aw6j22wp.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tmp15 = tmp14 + tmp8
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp6 * tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhm5nvr4ptrvercg475kt4xm2pewzxk75lvbbvze6jwnkcpxwzj.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznpaljv4oy7zvk2czop372ej4a4no4sy47d4vomc5mnwimzrgdi.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112) % 112
    x2 = (xindex // 12544)
    x3 = xindex % 12544
    x5 = xindex
    x6 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp7 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp19 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp31 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp37 = tl.load(in_ptr2 + (x5), None)
    tmp40 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
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
    tmp38 = tmp37 <= tmp4
    tmp39 = tl.where(tmp38, tmp4, tmp36)
    tmp41 = 1e-05
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp39 * tmp45
    tl.store(out_ptr0 + (x5), tmp36, None)
    tl.store(out_ptr1 + (x5), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xz/cxzyu5s2e36zycmmsedrb2kljs6mphwzeogo6bbyqsio3jj4omau.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (802816*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (802816*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (802816*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qu/cqumfti324jeawir6os6nass47nd2ozif3ffsq6xj4sgrhtqysrs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdrz6dk2pzt7ckgtogjngq3x2oujvjdyuzsixyvgyh3dxrzuvx6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, convolution, relu, getitem, getitem_1, convolution_1, relu_1, convolution_2, relu_2, convolution_3, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, relu_6, convolution_8, relu_7, convolution_9, relu_8, convolution_10, relu_9, convolution_11, relu_10, convolution_12, relu_11, convolution_13, convolution_14, relu_12, convolution_15, relu_13, convolution_16, relu_14, convolution_17, relu_15, convolution_18, relu_16, convolution_19, relu_17, convolution_20, relu_18, convolution_21, relu_19, convolution_22, relu_20, convolution_23, relu_21, convolution_24, relu_22, convolution_25, relu_23, convolution_26, convolution_27, relu_24, convolution_28, relu_25, convolution_29, relu_26, convolution_30, relu_27, convolution_31, relu_28, convolution_32, relu_29, convolution_33, relu_30, convolution_34, relu_31, convolution_35, relu_32, convolution_36, relu_33, convolution_37, relu_34, convolution_38, relu_35, convolution_39, relu_36, convolution_40, relu_37, convolution_41, relu_38, convolution_42, relu_39, convolution_43, relu_40, convolution_44, relu_41, convolution_45, convolution_46, relu_42, convolution_47, relu_43, convolution_48, relu_44, convolution_49, relu_45, convolution_50, relu_46, convolution_51, relu_47, convolution_52, view, permute_1, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_10, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_16, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_22, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_28, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_40, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_52, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_58, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_61, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_64, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_67, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_70, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_76, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_77, (512, ), (1, ))
    assert_size_stride(primals_79, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_85, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_88, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_91, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_94, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_97, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_100, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_103, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_106, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_109, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_112, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_115, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_118, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_121, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_124, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_127, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_136, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_142, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (1024, ), (1, ))
    assert_size_stride(primals_145, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_148, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_151, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (1024, ), (1, ))
    assert_size_stride(primals_154, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_157, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_237, (512, ), (1, ))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (512, ), (1, ))
    assert_size_stride(primals_249, (512, ), (1, ))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, ), (1, ))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (512, ), (1, ))
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (512, ), (1, ))
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_282, (512, ), (1, ))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_285, (512, ), (1, ))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_291, (1024, ), (1, ))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_294, (1024, ), (1, ))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_297, (2048, ), (1, ))
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (1024, ), (1, ))
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_310, (2048, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (1024, ), (1, ))
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(primals_319, (2048, ), (1, ))
    assert_size_stride(primals_321, (4, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(relu, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(getitem, (4, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(getitem_1, (4, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_1, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_1, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_2, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_2, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_3, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_4, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(relu_3, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_5, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_4, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_6, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_5, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_7, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(relu_6, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_8, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_7, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_9, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_8, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_10, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(relu_9, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_11, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(relu_10, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_12, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_11, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_13, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_14, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_12, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_15, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_13, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_16, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_14, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_17, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_15, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_18, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_16, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_19, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_17, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_20, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_18, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_21, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_19, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_22, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(relu_20, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_23, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_21, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_24, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_22, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_25, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_23, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_26, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_27, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_24, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_28, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_25, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_29, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_26, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_30, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_27, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_31, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_28, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_32, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_29, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_33, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_30, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_34, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_31, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_35, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_32, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_36, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_33, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_37, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_34, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_38, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_35, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_39, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_36, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_40, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_37, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_41, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(relu_38, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_42, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_39, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_43, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(relu_40, (4, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_44, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(relu_41, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_45, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(convolution_46, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(relu_42, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(convolution_47, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(relu_43, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_48, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(relu_44, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_49, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(relu_45, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(convolution_50, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(relu_46, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_51, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(relu_47, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_52, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(view, (4, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (4, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), view, out=buf1)
        del view
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf6 = empty((4, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, primals_319, primals_158, buf6, 401408, grid=grid(401408), stream=stream0)
        del primals_158
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_157
        buf8 = buf7[0]
        buf13 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2.run(relu_47, buf8, primals_316, primals_155, buf13, 200704, grid=grid(200704), stream=stream0)
        del primals_155
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf14 = aten.convolution_backward(buf13, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_154
        buf15 = buf14[0]
        buf20 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2.run(relu_46, buf15, primals_313, primals_152, buf20, 200704, grid=grid(200704), stream=stream0)
        del primals_152
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf21 = aten.convolution_backward(buf20, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf20
        del primals_151
        buf22 = buf21[0]
        buf3 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf4 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf24 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf25 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf5 = buf4; del buf4  # reuse
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_3.run(buf5, buf26, le, buf0, convolution_52, primals_318, relu_45, buf22, convolution_49, primals_309, primals_319, primals_310, buf3, buf24, 2048, 196, grid=grid(2048), stream=stream0)
        del convolution_49
        del convolution_52
        del primals_309
        del primals_318
        del primals_319
        buf9 = buf7[1]
        del buf7
        buf10 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf11 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf12 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(buf12, relu_47, buf8, convolution_51, primals_315, primals_316, buf10, 1024, 196, grid=grid(1024), stream=stream0)
        del buf8
        del convolution_51
        del primals_315
        del primals_316
        del relu_47
        buf16 = buf14[1]
        del buf14
        buf17 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf18 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(buf19, relu_46, buf15, convolution_50, primals_312, primals_313, buf17, 1024, 196, grid=grid(1024), stream=stream0)
        del buf15
        del convolution_50
        del primals_312
        del primals_313
        del relu_46
        buf23 = buf21[1]
        del buf21
        buf27 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_5.run(relu_45, le, buf0, buf22, primals_310, primals_149, buf27, 401408, grid=grid(401408), stream=stream0)
        del primals_149
        del primals_310
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf28 = aten.convolution_backward(buf27, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_148
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf32 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(buf33, relu_44, buf29, convolution_48, primals_306, primals_307, buf31, 1024, 196, grid=grid(1024), stream=stream0)
        del convolution_48
        del primals_306
        buf34 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(buf34, relu_44, primals_307, primals_146, 200704, grid=grid(200704), stream=stream0)
        del primals_146
        del primals_307
        del relu_44
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf35 = aten.convolution_backward(buf34, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf34
        del primals_145
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf39 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf40 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(buf40, relu_43, buf36, convolution_47, primals_303, primals_304, buf38, 1024, 196, grid=grid(1024), stream=stream0)
        del convolution_47
        del primals_303
        buf41 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(buf41, relu_43, primals_304, primals_143, 200704, grid=grid(200704), stream=stream0)
        del primals_143
        del primals_304
        del relu_43
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf42 = aten.convolution_backward(buf41, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf41
        del primals_142
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf45 = buf22; del buf22  # reuse
        buf49 = buf27; del buf27  # reuse
        buf55 = empty((4, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_7.run(buf45, relu_42, relu_45, le, buf0, buf43, primals_301, primals_140, primals_298, primals_137, buf49, buf55, 401408, grid=grid(401408), stream=stream0)
        del buf0
        del buf43
        del le
        del primals_137
        del primals_140
        del relu_42
        del relu_45
        buf46 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf47 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf53 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf48 = buf47; del buf47  # reuse
        buf54 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_8.run(buf48, buf54, buf45, convolution_46, primals_300, convolution_45, primals_297, primals_301, primals_298, buf46, 2048, 196, grid=grid(2048), stream=stream0)
        del buf45
        del convolution_45
        del convolution_46
        del primals_297
        del primals_298
        del primals_300
        del primals_301
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf50 = aten.convolution_backward(buf49, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf49
        del primals_139
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf56 = aten.convolution_backward(buf55, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf55
        del primals_136
        buf57 = buf56[0]
        buf58 = buf56[1]
        del buf56
        buf59 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf60 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_4.run(buf61, relu_41, buf57, convolution_44, primals_294, primals_295, buf59, 1024, 196, grid=grid(1024), stream=stream0)
        del convolution_44
        del primals_294
        buf62 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(buf62, relu_41, primals_295, primals_134, 200704, grid=grid(200704), stream=stream0)
        del primals_134
        del primals_295
        del relu_41
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf63 = aten.convolution_backward(buf62, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf62
        del primals_133
        buf64 = buf63[0]
        buf65 = buf63[1]
        del buf63
        buf66 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf67 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf68 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf68, relu_40, buf64, convolution_43, primals_291, primals_292, buf66, 1024, 784, grid=grid(1024), stream=stream0)
        del convolution_43
        del primals_291
        buf69 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf69, relu_40, primals_292, primals_131, 802816, grid=grid(802816), stream=stream0)
        del primals_131
        del primals_292
        del relu_40
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf70 = aten.convolution_backward(buf69, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_130
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf74 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_11.run(buf75, relu_39, buf51, buf71, convolution_42, primals_288, primals_289, buf73, 1024, 784, grid=grid(1024), stream=stream0)
        del convolution_42
        del primals_288
        buf76 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_39, buf51, buf71, primals_289, primals_128, buf76, 802816, grid=grid(802816), stream=stream0)
        del primals_128
        del primals_289
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf77 = aten.convolution_backward(buf76, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_127
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = empty((512, ), device='cuda', dtype=torch.float32)
        buf81 = empty((512, ), device='cuda', dtype=torch.float32)
        buf82 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf82, relu_38, buf78, convolution_41, primals_285, primals_286, buf80, 512, 784, grid=grid(512), stream=stream0)
        del convolution_41
        del primals_285
        buf83 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf83, relu_38, primals_286, primals_125, 401408, grid=grid(401408), stream=stream0)
        del primals_125
        del primals_286
        del relu_38
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf84 = aten.convolution_backward(buf83, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf83
        del primals_124
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf87 = empty((512, ), device='cuda', dtype=torch.float32)
        buf88 = empty((512, ), device='cuda', dtype=torch.float32)
        buf89 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf89, relu_37, buf85, convolution_40, primals_282, primals_283, buf87, 512, 784, grid=grid(512), stream=stream0)
        del convolution_40
        del primals_282
        buf90 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf90, relu_37, primals_283, primals_122, 401408, grid=grid(401408), stream=stream0)
        del primals_122
        del primals_283
        del relu_37
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf91 = aten.convolution_backward(buf90, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_121
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        buf94 = buf51; del buf51  # reuse
        buf98 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf94, relu_36, relu_39, buf71, buf92, primals_280, primals_119, buf98, 802816, grid=grid(802816), stream=stream0)
        del buf71
        del buf92
        del primals_119
        del relu_36
        del relu_39
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf99 = aten.convolution_backward(buf98, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_118
        buf100 = buf99[0]
        buf105 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16.run(relu_35, buf100, primals_277, primals_116, buf105, 401408, grid=grid(401408), stream=stream0)
        del primals_116
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf106 = aten.convolution_backward(buf105, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_115
        buf107 = buf106[0]
        buf112 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16.run(relu_34, buf107, primals_274, primals_113, buf112, 401408, grid=grid(401408), stream=stream0)
        del primals_113
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf113 = aten.convolution_backward(buf112, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf112
        del primals_112
        buf114 = buf113[0]
        buf95 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf96 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf116 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf117 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf97 = buf96; del buf96  # reuse
        buf118 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_17.run(buf97, buf118, buf94, convolution_39, primals_279, relu_33, buf114, convolution_36, primals_270, primals_280, primals_271, buf95, buf116, 1024, 784, grid=grid(1024), stream=stream0)
        del convolution_36
        del convolution_39
        del primals_270
        del primals_279
        del primals_280
        buf101 = buf99[1]
        del buf99
        buf102 = empty((512, ), device='cuda', dtype=torch.float32)
        buf103 = empty((512, ), device='cuda', dtype=torch.float32)
        buf104 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf104, relu_35, buf100, convolution_38, primals_276, primals_277, buf102, 512, 784, grid=grid(512), stream=stream0)
        del buf100
        del convolution_38
        del primals_276
        del primals_277
        del relu_35
        buf108 = buf106[1]
        del buf106
        buf109 = empty((512, ), device='cuda', dtype=torch.float32)
        buf110 = empty((512, ), device='cuda', dtype=torch.float32)
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf111, relu_34, buf107, convolution_37, primals_273, primals_274, buf109, 512, 784, grid=grid(512), stream=stream0)
        del buf107
        del convolution_37
        del primals_273
        del primals_274
        del relu_34
        buf115 = buf113[1]
        del buf113
        buf119 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_33, buf94, buf114, primals_271, primals_110, buf119, 802816, grid=grid(802816), stream=stream0)
        del primals_110
        del primals_271
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf120 = aten.convolution_backward(buf119, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_109
        buf121 = buf120[0]
        buf122 = buf120[1]
        del buf120
        buf123 = empty((512, ), device='cuda', dtype=torch.float32)
        buf124 = empty((512, ), device='cuda', dtype=torch.float32)
        buf125 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf125, relu_32, buf121, convolution_35, primals_267, primals_268, buf123, 512, 784, grid=grid(512), stream=stream0)
        del convolution_35
        del primals_267
        buf126 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf126, relu_32, primals_268, primals_107, 401408, grid=grid(401408), stream=stream0)
        del primals_107
        del primals_268
        del relu_32
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf127 = aten.convolution_backward(buf126, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf126
        del primals_106
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = empty((512, ), device='cuda', dtype=torch.float32)
        buf131 = empty((512, ), device='cuda', dtype=torch.float32)
        buf132 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf132, relu_31, buf128, convolution_34, primals_264, primals_265, buf130, 512, 784, grid=grid(512), stream=stream0)
        del convolution_34
        del primals_264
        buf133 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf133, relu_31, primals_265, primals_104, 401408, grid=grid(401408), stream=stream0)
        del primals_104
        del primals_265
        del relu_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf134 = aten.convolution_backward(buf133, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_103
        buf135 = buf134[0]
        buf136 = buf134[1]
        del buf134
        buf137 = buf114; del buf114  # reuse
        buf141 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(buf137, relu_30, relu_33, buf94, buf135, primals_262, primals_101, buf141, 802816, grid=grid(802816), stream=stream0)
        del buf135
        del primals_101
        del relu_30
        del relu_33
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf142 = aten.convolution_backward(buf141, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf143 = buf142[0]
        buf148 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16.run(relu_29, buf143, primals_259, primals_98, buf148, 401408, grid=grid(401408), stream=stream0)
        del primals_98
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf149 = aten.convolution_backward(buf148, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_97
        buf150 = buf149[0]
        buf155 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16.run(relu_28, buf150, primals_256, primals_95, buf155, 401408, grid=grid(401408), stream=stream0)
        del primals_95
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf156 = aten.convolution_backward(buf155, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf155
        del primals_94
        buf157 = buf156[0]
        buf138 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf139 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf159 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf160 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf140 = buf139; del buf139  # reuse
        buf161 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_17.run(buf140, buf161, buf137, convolution_33, primals_261, relu_27, buf157, convolution_30, primals_252, primals_262, primals_253, buf138, buf159, 1024, 784, grid=grid(1024), stream=stream0)
        del convolution_30
        del convolution_33
        del primals_252
        del primals_261
        del primals_262
        buf144 = buf142[1]
        del buf142
        buf145 = empty((512, ), device='cuda', dtype=torch.float32)
        buf146 = empty((512, ), device='cuda', dtype=torch.float32)
        buf147 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf147, relu_29, buf143, convolution_32, primals_258, primals_259, buf145, 512, 784, grid=grid(512), stream=stream0)
        del buf143
        del convolution_32
        del primals_258
        del primals_259
        del relu_29
        buf151 = buf149[1]
        del buf149
        buf152 = empty((512, ), device='cuda', dtype=torch.float32)
        buf153 = empty((512, ), device='cuda', dtype=torch.float32)
        buf154 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf154, relu_28, buf150, convolution_31, primals_255, primals_256, buf152, 512, 784, grid=grid(512), stream=stream0)
        del buf150
        del convolution_31
        del primals_255
        del primals_256
        del relu_28
        buf158 = buf156[1]
        del buf156
        buf162 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_27, buf137, buf157, primals_253, primals_92, buf162, 802816, grid=grid(802816), stream=stream0)
        del primals_253
        del primals_92
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf163 = aten.convolution_backward(buf162, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_91
        buf164 = buf163[0]
        buf165 = buf163[1]
        del buf163
        buf166 = empty((512, ), device='cuda', dtype=torch.float32)
        buf167 = empty((512, ), device='cuda', dtype=torch.float32)
        buf168 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf168, relu_26, buf164, convolution_29, primals_249, primals_250, buf166, 512, 784, grid=grid(512), stream=stream0)
        del convolution_29
        del primals_249
        buf169 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf169, relu_26, primals_250, primals_89, 401408, grid=grid(401408), stream=stream0)
        del primals_250
        del primals_89
        del relu_26
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf170 = aten.convolution_backward(buf169, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf169
        del primals_88
        buf171 = buf170[0]
        buf172 = buf170[1]
        del buf170
        buf173 = empty((512, ), device='cuda', dtype=torch.float32)
        buf174 = empty((512, ), device='cuda', dtype=torch.float32)
        buf175 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf175, relu_25, buf171, convolution_28, primals_246, primals_247, buf173, 512, 784, grid=grid(512), stream=stream0)
        del convolution_28
        del primals_246
        buf176 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf176, relu_25, primals_247, primals_86, 401408, grid=grid(401408), stream=stream0)
        del primals_247
        del primals_86
        del relu_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf177 = aten.convolution_backward(buf176, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf176
        del primals_85
        buf178 = buf177[0]
        buf179 = buf177[1]
        del buf177
        buf180 = buf137; del buf137  # reuse
        buf184 = buf162; del buf162  # reuse
        buf190 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf180, relu_24, relu_27, buf157, buf178, primals_244, primals_83, primals_241, primals_80, buf184, buf190, 802816, grid=grid(802816), stream=stream0)
        del buf157
        del buf178
        del primals_80
        del primals_83
        del relu_24
        del relu_27
        buf181 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf182 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf188 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf183 = buf182; del buf182  # reuse
        buf189 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_20.run(buf183, buf189, buf180, convolution_27, primals_243, convolution_26, primals_240, primals_244, primals_241, buf181, 1024, 784, grid=grid(1024), stream=stream0)
        del buf180
        del convolution_26
        del convolution_27
        del primals_240
        del primals_241
        del primals_243
        del primals_244
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf185 = aten.convolution_backward(buf184, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf184
        del primals_82
        buf186 = buf185[0]
        buf187 = buf185[1]
        del buf185
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf191 = aten.convolution_backward(buf190, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf190
        del primals_79
        buf192 = buf191[0]
        buf193 = buf191[1]
        del buf191
        buf194 = empty((512, ), device='cuda', dtype=torch.float32)
        buf195 = empty((512, ), device='cuda', dtype=torch.float32)
        buf196 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_13.run(buf196, relu_23, buf192, convolution_25, primals_237, primals_238, buf194, 512, 784, grid=grid(512), stream=stream0)
        del convolution_25
        del primals_237
        buf197 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf197, relu_23, primals_238, primals_77, 401408, grid=grid(401408), stream=stream0)
        del primals_238
        del primals_77
        del relu_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf198 = aten.convolution_backward(buf197, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf197
        del primals_76
        buf199 = buf198[0]
        buf200 = buf198[1]
        del buf198
        buf201 = empty((512, ), device='cuda', dtype=torch.float32)
        buf202 = empty((512, ), device='cuda', dtype=torch.float32)
        buf203 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_21.run(buf203, relu_22, buf199, convolution_24, primals_234, primals_235, buf201, 512, 3136, grid=grid(512), stream=stream0)
        del convolution_24
        del primals_234
        buf204 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22.run(buf204, relu_22, primals_235, primals_74, 1605632, grid=grid(1605632), stream=stream0)
        del primals_235
        del primals_74
        del relu_22
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf205 = aten.convolution_backward(buf204, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_73
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = empty((512, ), device='cuda', dtype=torch.float32)
        buf209 = empty((512, ), device='cuda', dtype=torch.float32)
        buf210 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf210, relu_21, buf186, buf206, convolution_23, primals_231, primals_232, buf208, 512, 3136, grid=grid(512), stream=stream0)
        del convolution_23
        del primals_231
        buf211 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(relu_21, buf186, buf206, primals_232, primals_71, buf211, 1605632, grid=grid(1605632), stream=stream0)
        del primals_232
        del primals_71
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf212 = aten.convolution_backward(buf211, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_70
        buf213 = buf212[0]
        buf214 = buf212[1]
        del buf212
        buf215 = empty((256, ), device='cuda', dtype=torch.float32)
        buf216 = empty((256, ), device='cuda', dtype=torch.float32)
        buf217 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf217, relu_20, buf213, convolution_22, primals_228, primals_229, buf215, 256, 3136, grid=grid(256), stream=stream0)
        del convolution_22
        del primals_228
        buf218 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(buf218, relu_20, primals_229, primals_68, 802816, grid=grid(802816), stream=stream0)
        del primals_229
        del primals_68
        del relu_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf219 = aten.convolution_backward(buf218, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf218
        del primals_67
        buf220 = buf219[0]
        buf221 = buf219[1]
        del buf219
        buf222 = empty((256, ), device='cuda', dtype=torch.float32)
        buf223 = empty((256, ), device='cuda', dtype=torch.float32)
        buf224 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf224, relu_19, buf220, convolution_21, primals_225, primals_226, buf222, 256, 3136, grid=grid(256), stream=stream0)
        del convolution_21
        del primals_225
        buf225 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(buf225, relu_19, primals_226, primals_65, 802816, grid=grid(802816), stream=stream0)
        del primals_226
        del primals_65
        del relu_19
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf226 = aten.convolution_backward(buf225, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_64
        buf227 = buf226[0]
        buf228 = buf226[1]
        del buf226
        buf229 = buf186; del buf186  # reuse
        buf233 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf229, relu_18, relu_21, buf206, buf227, primals_223, primals_62, buf233, 1605632, grid=grid(1605632), stream=stream0)
        del buf206
        del primals_62
        del relu_18
        del relu_21
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf234 = aten.convolution_backward(buf233, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        buf235 = buf234[0]
        buf240 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(relu_17, buf235, primals_220, primals_59, buf240, 802816, grid=grid(802816), stream=stream0)
        del primals_59
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf241 = aten.convolution_backward(buf240, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_58
        buf242 = buf241[0]
        buf247 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(relu_16, buf242, primals_217, primals_56, buf247, 802816, grid=grid(802816), stream=stream0)
        del primals_56
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf248 = aten.convolution_backward(buf247, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf247
        del primals_55
        buf249 = buf248[0]
        buf230 = empty((512, ), device='cuda', dtype=torch.float32)
        buf231 = empty((512, ), device='cuda', dtype=torch.float32)
        buf251 = empty((512, ), device='cuda', dtype=torch.float32)
        buf252 = empty((512, ), device='cuda', dtype=torch.float32)
        buf232 = buf231; del buf231  # reuse
        buf253 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_29.run(buf232, buf253, buf229, convolution_20, primals_222, relu_15, buf249, convolution_17, primals_213, primals_223, primals_214, buf230, buf251, 512, 3136, grid=grid(512), stream=stream0)
        del convolution_17
        del convolution_20
        del primals_213
        del primals_222
        del primals_223
        buf236 = buf234[1]
        del buf234
        buf237 = empty((256, ), device='cuda', dtype=torch.float32)
        buf238 = empty((256, ), device='cuda', dtype=torch.float32)
        buf239 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf239, relu_17, buf235, convolution_19, primals_219, primals_220, buf237, 256, 3136, grid=grid(256), stream=stream0)
        del buf235
        del convolution_19
        del primals_219
        del primals_220
        del relu_17
        buf243 = buf241[1]
        del buf241
        buf244 = empty((256, ), device='cuda', dtype=torch.float32)
        buf245 = empty((256, ), device='cuda', dtype=torch.float32)
        buf246 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf246, relu_16, buf242, convolution_18, primals_216, primals_217, buf244, 256, 3136, grid=grid(256), stream=stream0)
        del buf242
        del convolution_18
        del primals_216
        del primals_217
        del relu_16
        buf250 = buf248[1]
        del buf248
        buf254 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(relu_15, buf229, buf249, primals_214, primals_53, buf254, 1605632, grid=grid(1605632), stream=stream0)
        del primals_214
        del primals_53
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf255 = aten.convolution_backward(buf254, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_52
        buf256 = buf255[0]
        buf257 = buf255[1]
        del buf255
        buf258 = empty((256, ), device='cuda', dtype=torch.float32)
        buf259 = empty((256, ), device='cuda', dtype=torch.float32)
        buf260 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf260, relu_14, buf256, convolution_16, primals_210, primals_211, buf258, 256, 3136, grid=grid(256), stream=stream0)
        del convolution_16
        del primals_210
        buf261 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(buf261, relu_14, primals_211, primals_50, 802816, grid=grid(802816), stream=stream0)
        del primals_211
        del primals_50
        del relu_14
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf262 = aten.convolution_backward(buf261, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf261
        del primals_49
        buf263 = buf262[0]
        buf264 = buf262[1]
        del buf262
        buf265 = empty((256, ), device='cuda', dtype=torch.float32)
        buf266 = empty((256, ), device='cuda', dtype=torch.float32)
        buf267 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf267, relu_13, buf263, convolution_15, primals_207, primals_208, buf265, 256, 3136, grid=grid(256), stream=stream0)
        del convolution_15
        del primals_207
        buf268 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(buf268, relu_13, primals_208, primals_47, 802816, grid=grid(802816), stream=stream0)
        del primals_208
        del primals_47
        del relu_13
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf269 = aten.convolution_backward(buf268, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf268
        del primals_46
        buf270 = buf269[0]
        buf271 = buf269[1]
        del buf269
        buf272 = buf229; del buf229  # reuse
        buf276 = buf254; del buf254  # reuse
        buf282 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30.run(buf272, relu_12, relu_15, buf249, buf270, primals_205, primals_44, primals_202, primals_41, buf276, buf282, 1605632, grid=grid(1605632), stream=stream0)
        del buf249
        del buf270
        del primals_41
        del primals_44
        del relu_12
        del relu_15
        buf273 = empty((512, ), device='cuda', dtype=torch.float32)
        buf274 = empty((512, ), device='cuda', dtype=torch.float32)
        buf280 = empty((512, ), device='cuda', dtype=torch.float32)
        buf275 = buf274; del buf274  # reuse
        buf281 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf275, buf281, buf272, convolution_14, primals_204, convolution_13, primals_201, primals_205, primals_202, buf273, 512, 3136, grid=grid(512), stream=stream0)
        del buf272
        del convolution_13
        del convolution_14
        del primals_201
        del primals_202
        del primals_204
        del primals_205
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf277 = aten.convolution_backward(buf276, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf276
        del primals_43
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf283 = aten.convolution_backward(buf282, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf282
        del primals_40
        buf284 = buf283[0]
        buf285 = buf283[1]
        del buf283
        buf286 = empty((256, ), device='cuda', dtype=torch.float32)
        buf287 = empty((256, ), device='cuda', dtype=torch.float32)
        buf288 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(buf288, relu_11, buf284, convolution_12, primals_198, primals_199, buf286, 256, 3136, grid=grid(256), stream=stream0)
        del convolution_12
        del primals_198
        buf289 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(buf289, relu_11, primals_199, primals_38, 802816, grid=grid(802816), stream=stream0)
        del primals_199
        del primals_38
        del relu_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf290 = aten.convolution_backward(buf289, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf289
        del primals_37
        buf291 = buf290[0]
        buf292 = buf290[1]
        del buf290
        buf293 = empty((256, ), device='cuda', dtype=torch.float32)
        buf294 = empty((256, ), device='cuda', dtype=torch.float32)
        buf295 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_32.run(buf295, relu_10, buf291, convolution_11, primals_195, primals_196, buf293, 256, 12544, grid=grid(256), stream=stream0)
        del convolution_11
        del primals_195
        buf296 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33.run(buf296, relu_10, primals_196, primals_35, 3211264, grid=grid(3211264), stream=stream0)
        del primals_196
        del primals_35
        del relu_10
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf297 = aten.convolution_backward(buf296, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_34
        buf298 = buf297[0]
        buf299 = buf297[1]
        del buf297
        buf300 = empty((256, ), device='cuda', dtype=torch.float32)
        buf301 = empty((256, ), device='cuda', dtype=torch.float32)
        buf302 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_34.run(buf302, relu_9, buf278, buf298, convolution_10, primals_192, primals_193, buf300, 256, 12544, grid=grid(256), stream=stream0)
        del convolution_10
        del primals_192
        buf303 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_35.run(relu_9, buf278, buf298, primals_193, primals_32, buf303, 3211264, grid=grid(3211264), stream=stream0)
        del primals_193
        del primals_32
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf304 = aten.convolution_backward(buf303, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_31
        buf305 = buf304[0]
        buf306 = buf304[1]
        del buf304
        buf307 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        buf309 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_8, buf305, convolution_9, primals_189, buf307, buf309, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_9
        del primals_189
        buf308 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf307, buf308, 128, 2, grid=grid(128), stream=stream0)
        buf310 = empty((128, ), device='cuda', dtype=torch.float32)
        buf311 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf311, buf309, primals_190, 128, 2, grid=grid(128), stream=stream0)
        buf312 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(buf312, relu_8, primals_190, primals_29, 1605632, grid=grid(1605632), stream=stream0)
        del primals_190
        del primals_29
        del relu_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf313 = aten.convolution_backward(buf312, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf312
        del primals_28
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = buf309; del buf309  # reuse
        buf318 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_7, buf314, convolution_8, primals_186, buf316, buf318, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_8
        del primals_186
        buf317 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf316, buf317, 128, 2, grid=grid(128), stream=stream0)
        buf319 = empty((128, ), device='cuda', dtype=torch.float32)
        buf320 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf320, buf318, primals_187, 128, 2, grid=grid(128), stream=stream0)
        buf321 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(buf321, relu_7, primals_187, primals_26, 1605632, grid=grid(1605632), stream=stream0)
        del primals_187
        del primals_26
        del relu_7
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf322 = aten.convolution_backward(buf321, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf323 = buf322[0]
        buf324 = buf322[1]
        del buf322
        buf325 = buf278; del buf278  # reuse
        buf329 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(buf325, relu_6, relu_9, buf298, buf323, primals_184, primals_23, buf329, 3211264, grid=grid(3211264), stream=stream0)
        del buf298
        del primals_23
        del relu_6
        del relu_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf330 = aten.convolution_backward(buf329, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf331 = buf330[0]
        buf338 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41.run(relu_5, buf331, primals_181, primals_20, buf338, 1605632, grid=grid(1605632), stream=stream0)
        del primals_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf339 = aten.convolution_backward(buf338, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_19
        buf340 = buf339[0]
        buf347 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41.run(relu_4, buf340, primals_178, primals_17, buf347, 1605632, grid=grid(1605632), stream=stream0)
        del primals_17
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf348 = aten.convolution_backward(buf347, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf347
        del primals_16
        buf349 = buf348[0]
        buf326 = reinterpret_tensor(buf318, (256, ), (1, ), 0); del buf318  # reuse
        buf327 = reinterpret_tensor(buf316, (256, ), (1, ), 0); del buf316  # reuse
        buf351 = empty((256, ), device='cuda', dtype=torch.float32)
        buf352 = empty((256, ), device='cuda', dtype=torch.float32)
        buf358 = empty((256, ), device='cuda', dtype=torch.float32)
        buf328 = buf327; del buf327  # reuse
        buf353 = buf352; del buf352  # reuse
        buf359 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_42.run(buf328, buf353, buf359, buf325, convolution_7, primals_183, relu_3, buf349, convolution_4, primals_174, convolution_3, primals_171, primals_184, primals_175, primals_172, buf326, buf351, 256, 12544, grid=grid(256), stream=stream0)
        del convolution_3
        del convolution_4
        del convolution_7
        del primals_171
        del primals_174
        del primals_183
        del primals_184
        buf332 = buf330[1]
        del buf330
        buf333 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        buf335 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_5, buf331, convolution_6, primals_180, buf333, buf335, 256, 6272, grid=grid(256), stream=stream0)
        del buf331
        del convolution_6
        del primals_180
        del relu_5
        buf334 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf333, buf334, 128, 2, grid=grid(128), stream=stream0)
        buf336 = empty((128, ), device='cuda', dtype=torch.float32)
        buf337 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf337, buf335, primals_181, 128, 2, grid=grid(128), stream=stream0)
        del primals_181
        buf341 = buf339[1]
        del buf339
        buf342 = buf335; del buf335  # reuse
        buf344 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_4, buf340, convolution_5, primals_177, buf342, buf344, 256, 6272, grid=grid(256), stream=stream0)
        del buf340
        del convolution_5
        del primals_177
        del relu_4
        buf343 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf342, buf343, 128, 2, grid=grid(128), stream=stream0)
        buf345 = empty((128, ), device='cuda', dtype=torch.float32)
        buf346 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf346, buf344, primals_178, 128, 2, grid=grid(128), stream=stream0)
        del primals_178
        buf350 = buf348[1]
        del buf348
        buf354 = buf329; del buf329  # reuse
        buf360 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(relu_3, buf325, buf349, primals_175, primals_14, primals_172, primals_11, buf354, buf360, 3211264, grid=grid(3211264), stream=stream0)
        del buf325
        del buf349
        del primals_11
        del primals_14
        del primals_172
        del primals_175
        del relu_3
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf355 = aten.convolution_backward(buf354, getitem, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_13
        buf356 = buf355[0]
        buf357 = buf355[1]
        del buf355
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf361 = aten.convolution_backward(buf360, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_10
        buf362 = buf361[0]
        buf363 = buf361[1]
        del buf361
        buf364 = buf344; del buf344  # reuse
        buf366 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_2, buf362, convolution_2, primals_168, buf364, buf366, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_2
        del primals_168
        buf365 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf364, buf365, 128, 2, grid=grid(128), stream=stream0)
        buf367 = empty((128, ), device='cuda', dtype=torch.float32)
        buf368 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf368, buf366, primals_169, 128, 2, grid=grid(128), stream=stream0)
        buf369 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(buf369, relu_2, primals_169, primals_8, 1605632, grid=grid(1605632), stream=stream0)
        del primals_169
        del primals_8
        del relu_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf370 = aten.convolution_backward(buf369, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf369
        del primals_7
        buf371 = buf370[0]
        buf372 = buf370[1]
        del buf370
        buf373 = buf366; del buf366  # reuse
        buf375 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_1, buf371, convolution_1, primals_165, buf373, buf375, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_1
        del primals_165
        buf374 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf373, buf374, 128, 2, grid=grid(128), stream=stream0)
        del buf373
        buf376 = empty((128, ), device='cuda', dtype=torch.float32)
        buf377 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf377, buf375, primals_166, 128, 2, grid=grid(128), stream=stream0)
        del buf375
        buf378 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(buf378, relu_1, primals_166, primals_5, 1605632, grid=grid(1605632), stream=stream0)
        del primals_166
        del primals_5
        del relu_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf379 = aten.convolution_backward(buf378, getitem, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf378
        del getitem
        del primals_4
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf382, buf380, 802816, grid=grid(802816), stream=stream0)
        del buf380
        buf383 = reinterpret_tensor(buf360, (4, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf360  # reuse
        buf389 = reinterpret_tensor(buf354, (4, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_45.run(getitem_1, buf382, relu, primals_163, primals_2, buf383, buf389, 3211264, grid=grid(3211264), stream=stream0)
        del buf382
        del getitem_1
        del primals_2
        buf384 = empty((64, 7), device='cuda', dtype=torch.float32)
        buf386 = empty((64, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu, buf383, convolution, primals_162, buf384, buf386, 448, 7168, grid=grid(448), stream=stream0)
        del buf383
        del convolution
        del primals_162
        del relu
        buf385 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf384, buf385, 64, 7, grid=grid(64), stream=stream0)
        del buf384
        buf387 = empty((64, ), device='cuda', dtype=torch.float32)
        buf388 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf388, buf386, primals_163, 64, 7, grid=grid(64), stream=stream0)
        del buf386
        del primals_163
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf390 = aten.convolution_backward(buf389, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf389
        del primals_1
        del primals_321
        buf391 = buf390[1]
        return (buf391, buf388, buf385, buf381, buf377, buf374, buf372, buf368, buf365, buf363, buf359, buf351, buf357, buf353, buf351, buf350, buf346, buf343, buf341, buf337, buf334, buf332, buf328, buf326, buf324, buf320, buf317, buf315, buf311, buf308, buf306, buf302, buf300, buf299, buf295, buf293, buf292, buf288, buf286, buf285, buf281, buf273, buf279, buf275, buf273, buf271, buf267, buf265, buf264, buf260, buf258, buf257, buf253, buf251, buf250, buf246, buf244, buf243, buf239, buf237, buf236, buf232, buf230, buf228, buf224, buf222, buf221, buf217, buf215, buf214, buf210, buf208, buf207, buf203, buf201, buf200, buf196, buf194, buf193, buf189, buf181, buf187, buf183, buf181, buf179, buf175, buf173, buf172, buf168, buf166, buf165, buf161, buf159, buf158, buf154, buf152, buf151, buf147, buf145, buf144, buf140, buf138, buf136, buf132, buf130, buf129, buf125, buf123, buf122, buf118, buf116, buf115, buf111, buf109, buf108, buf104, buf102, buf101, buf97, buf95, buf93, buf89, buf87, buf86, buf82, buf80, buf79, buf75, buf73, buf72, buf68, buf66, buf65, buf61, buf59, buf58, buf54, buf46, buf52, buf48, buf46, buf44, buf40, buf38, buf37, buf33, buf31, buf30, buf26, buf24, buf23, buf19, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((4, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.int64)
    convolution_1 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((4, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, convolution, relu, getitem, getitem_1, convolution_1, relu_1, convolution_2, relu_2, convolution_3, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, relu_6, convolution_8, relu_7, convolution_9, relu_8, convolution_10, relu_9, convolution_11, relu_10, convolution_12, relu_11, convolution_13, convolution_14, relu_12, convolution_15, relu_13, convolution_16, relu_14, convolution_17, relu_15, convolution_18, relu_16, convolution_19, relu_17, convolution_20, relu_18, convolution_21, relu_19, convolution_22, relu_20, convolution_23, relu_21, convolution_24, relu_22, convolution_25, relu_23, convolution_26, convolution_27, relu_24, convolution_28, relu_25, convolution_29, relu_26, convolution_30, relu_27, convolution_31, relu_28, convolution_32, relu_29, convolution_33, relu_30, convolution_34, relu_31, convolution_35, relu_32, convolution_36, relu_33, convolution_37, relu_34, convolution_38, relu_35, convolution_39, relu_36, convolution_40, relu_37, convolution_41, relu_38, convolution_42, relu_39, convolution_43, relu_40, convolution_44, relu_41, convolution_45, convolution_46, relu_42, convolution_47, relu_43, convolution_48, relu_44, convolution_49, relu_45, convolution_50, relu_46, convolution_51, relu_47, convolution_52, view, permute_1, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnext50_32x4d', benchmark_compiled_module)
