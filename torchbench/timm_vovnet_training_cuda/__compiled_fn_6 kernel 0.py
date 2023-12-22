
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcfzxopkuatqil5htk4wi7d4gvu4y72chxfxrjmw2vntq5r6g34.py
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
    size_hints=[1024, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp17 * tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckubuikbu7jmxnwlcfp2gw5vhuzxszbu74usiuwwpwrzspykn5vz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjdxh7a4s7q6h6nes7townqemhj6so4ajrjfl2uqex2faz2bjht.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (94080 + r1 + (49*x0) + (105056*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp15 * tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coie3k4hrpklq6nzjthpbzvwetcbdm7cglcbv2njlohuisxtkgnh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (94080 + x4 + (105056*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7hbnj4lpm3cb2se3qnw3cilir5ddcui4v3k2x4zwwzcezv7rt2t.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (83104 + r1 + (49*x0) + (105056*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4m/c4mfa4zyugwaw6n3trxu66ephg3bbfjvr4zttgju3zlo4mwkdugt.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (83104 + x4 + (105056*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqtnycuc3hf72szxbqsmicsypxhd2l4dcdkjuhywzinwlafoht6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (72128 + r1 + (49*x0) + (105056*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdoqivlygsbdxri2mrmndy5zy2nev6tbm65flxrze33wktraolxh.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (72128 + x4 + (105056*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlzzox3uhbkrr5z4s2bvhj676ylzahhq7dged7snhk2nvzy6cqs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (61152 + r1 + (49*x0) + (105056*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgts7rkrhcscv42zemom7m5t7rpxgu7igz5uzkzrta25ssfejk2j.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (61152 + x4 + (105056*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2muww3yq7sczge7gy65mmopgcymbpuax5htx4uut2od4dmvpht.py
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
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (50176 + r1 + (49*x0) + (105056*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4likybn6c235fnujoqkxnctw6t2g27w7ruiy6rzhovnfkakog7f.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (50176 + x4 + (105056*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sj/csjac2dufnwq5nys54rvhnxpkmaukgf2abj2slaj2pj54edsagrd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (105056*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2cbqtaptwn6tqbr64vntoofiw4yytupndavzeiwswsmev5w6fk.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x4 + (105056*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c755xzmvnufszimbbciv6xqmvxol2ep363sxaxvyoerv3usv6sod.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (81536 + r1 + (49*x0) + (92512*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp15 * tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijsbkrsthjy3gvqvezgwn6lrpxmft535skqpvyikz4zwtl66n2y.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (81536 + x4 + (92512*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz42m2aaj7vtzeoswxybhtsorarqlgisa22uasjh7ub5l37unjay.py
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
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (70560 + r1 + (49*x0) + (92512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwiozgfzrmb2kfvmyfxuoyj5kj5vwk53utochupevq3mawqkyo6x.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (70560 + x4 + (92512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbghfsezzit4l2i6utifunj3fl2kc4h4hzf45kkkxxtomdood4gs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (59584 + r1 + (49*x0) + (92512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbnbcmgs5v65estbkhoqfxvpd6u46ve6svjpcb3aphwmvnst44z.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (59584 + x4 + (92512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbjm3v63kluqbsx24uds6ca52ewlujgpbr3xgxkuudnwyszuo4q.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (48608 + r1 + (49*x0) + (92512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdq3k3qlo42chsigctks6h6xt75urys2xwta2enqrldn5vheslpo.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (48608 + x4 + (92512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czdgjumnmmeh2lefrn4dfgpdeg4l7oeltznoy2wrblmjxorivmuq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (37632 + r1 + (49*x0) + (92512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j4/cj45zqlszjd6bs33jhy347uungragjwztqxgly56ehzx2fqu34yg.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 10976)
    x4 = xindex % 10976
    x1 = (xindex // 49) % 224
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (37632 + x4 + (92512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctu6p7iha6xokfdxwjdiiaabhvk7e6jf5o3iqvmsuiqfuv6brzmt.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 37632
    x1 = (xindex // 37632)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (92512*x1)), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctfpbv2gzaau3ulickhjbjdv36g4n7ojjkynpe324ssnkk6i2rj.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196)
    x3 = xindex % 196
    x5 = xindex
    x6 = (xindex // 196) % 768
    tmp0 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((7*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + ((7*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(7, 1 + (x1 // 2))))) >= 0, 0, 7))) + (49*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(7, 1 + (x0 // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr2 + (x5), None)
    tmp40 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (((-1) + x1) // 2))
    tmp10 = tl.math.min(7, 1 + (x1 // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (((-1) + x0) // 2)))
    tmp13 = tl.math.min(7, 1 + (x0 // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (((-1) + x1) // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (((-1) + x0) // 2))
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


# kernel path: /tmp/torchinductor_youkaichao/oo/cooohyfrdsuxddh75ycv5bsbnje67tlv2plqh33xyht6xyqjabff.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tw6opepjjkxwnrl3hux3cnunidnqfvs27cojdn7ct3pq7przxe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (301056 + r1 + (196*x0) + (338688*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp15 * tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccver6dwiri7jevbxdp2bqlevohnrlrn3kbo7btyagogeywple6x.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (301056 + x4 + (338688*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymoe4qk2ptg35b4btftgieyd7hqftu7soylqayo2ifshkyvukbn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (263424 + r1 + (196*x0) + (338688*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ve/cveyr45mbdo2etq2uwb2bnf5fdfir5fjpuonpndwzg7kmjjfz2wp.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (263424 + x4 + (338688*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zyxkpwbykegvkkmi4zto3i7vcvqejb76gyipxvg2xv2dk55x2l.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (225792 + r1 + (196*x0) + (338688*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqxqnlhw7socqjtfwtw5wtlwoynli2susg2ql7cic3n7hobsbjj.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (225792 + x4 + (338688*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzurokunprs35ix4eyxqk77cgqqx4tcbeypzz34oedmaqhajqye.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (188160 + r1 + (196*x0) + (338688*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bc/cbc6qx2bkatzotxf6waw4rruzybapsq4r54y4fb3vy7eyz4vsbsm.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (188160 + x4 + (338688*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcqyosxvudpdiog7jfvpk6iey576lcjhlftzjcbpvlkoh2wuqny.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (150528 + r1 + (196*x0) + (338688*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ob/cob5ynnjnwxmzvzgj3eadlevhlcv7q6on3e3idod3ynrxu67zgr6.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (150528 + x4 + (338688*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/ckn7susiowiofg7ktxhs5tg7rztpi4dxpx5tt53cjjl6hip63vu7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (338688*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpp7p6kanzbxbd4bz4s54t5xdbmei6petrxwq42pexyfujya6zu.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 150528)
    x4 = xindex % 150528
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x4 + (338688*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c643rcuzy2pmbraj2wnymowkz43nmb6oljat6spnsbew6d3v6bxk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (250880 + r1 + (196*x0) + (288512*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp15 * tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3nqzhhlcrz6r3hd43ifpfuj7e5u33leaouaxdgjbaj76sihfri.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (250880 + x4 + (288512*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbkn7or753lrwf7dwkwzixdo22kj6cvbgsdxv2kbamv6gvbfzju.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (213248 + r1 + (196*x0) + (288512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hh/chhe5zfkiiyvgtrj3rx6i2ryf2keodjhtkk6gcqv23dsynpeaglc.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (213248 + x4 + (288512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4ioc7yuqmg4onskpx3weuosgni23lpnxa4clanlewozaq6t2xbn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (175616 + r1 + (196*x0) + (288512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5qai2bkg6sxqu627o5rwei2eybonbh5zmfjkc7k2ltqcrc4w4o.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (175616 + x4 + (288512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmg5iihxgbybvuhbfy47a4dsm5wjgxiq6gyf7ty6qw72gk3cjja.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (137984 + r1 + (196*x0) + (288512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fv/cfv2ywmrfy5mvohxewgwo3raergfcji5o4roylja6cbq3sgrr3ym.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (137984 + x4 + (288512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltbi47x7fzz2xb4gvqdc6rm6sop4uvbnpjt5tagtmqgc5jnwdrr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (100352 + r1 + (196*x0) + (288512*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zk/czkcoiinue76ycanfatr2bx5ubcnrrwej6rssjbm4sqdm4cygh4h.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x4 = xindex % 37632
    x1 = (xindex // 196) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (100352 + x4 + (288512*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pw/cpw5qmbqdxyxfudghvnerviln6l2fnm3zvvely6hk3pwuth6u4ct.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (288512*x1)), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvc4vvzwwyubsanw5uxolxvaycbpbqa2ggmmgosunqwxaqe4hxy.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x3 = xindex % 784
    x5 = xindex
    x6 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((14*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + ((14*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (196*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr2 + (x5), None)
    tmp40 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (((-1) + x1) // 2))
    tmp10 = tl.math.min(14, 1 + (x1 // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (((-1) + x0) // 2)))
    tmp13 = tl.math.min(14, 1 + (x0 // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (((-1) + x1) // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (((-1) + x0) // 2))
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


# kernel path: /tmp/torchinductor_youkaichao/5j/c5j4nwjbjph7dggh7qbes2v4hrgfuzf3fevswbmoi2zvgxzyig5w.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/56/c564vztaagrruy6ic4xnr65jc52sv7lphr32q2kepbyiuf2wxhig.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (702464 + r1 + (784*x0) + (827904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp12 * tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csokzwxnedej3aos25cslnq4pwyuuycedprgaqkoy6pzghtmnkqx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (702464 + x4 + (827904*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/clyoum5ghfy7x7oi25pv3z3cer4ng7nyj2x7dngnc7uxlpncrk2t.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (577024 + r1 + (784*x0) + (827904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwowpoh23m5fnuomzlb2fzqqncpbfw33uovshbow7tlpz5cc2fd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (577024 + x4 + (827904*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gp/cgpck675c3ji2kjty4yedrfpc3v65vnt32b7jgox7gysm355is3v.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (451584 + r1 + (784*x0) + (827904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bf/cbf6u3ahifdbci6rrstj7j5fwkdc4j3ozo3crv6yc5n7zhzizxcb.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (451584 + x4 + (827904*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsazcwtfwa5pttxcsn4ngnyeb3xjsvkiriwlnzqzo62cuhx7gt5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (326144 + r1 + (784*x0) + (827904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ec/cece2ymp32yaizqbxooemadlh7zvviv5lqycplzjo5qaswr4z35m.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (326144 + x4 + (827904*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujry2a7grwwzzgnmpxxvmq5rhleo3rtahtqyer3hhdzuzbvqumz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + r1 + (784*x0) + (827904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/d3/cd36sr6wp55hhpaigdvahvwjm2seqdv253y6shhekkdyyyx3g643.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 125440)
    x4 = xindex % 125440
    x1 = (xindex // 784) % 160
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (200704 + x4 + (827904*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqzpe3hrqvdqvvp3a2yfyy77lv554zfzgdcnf6lyd5ccn6w25y3.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (827904*x1)), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkr7wkx6rr5qei2qj4zg52gkzjnhwtnyivym4xt25ufytkwxine.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    x3 = xindex % 3136
    x5 = xindex
    x6 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + ((28*(tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (((-1) + x1) // 2)), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (((-1) + x0) // 2)), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + ((28*(tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x1) // 2))), (-1) + (tl.math.min(28, 1 + (x1 // 2))))) >= 0, 0, 28))) + (784*x2) + (tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (((-1) + x0) // 2))), (-1) + (tl.math.min(28, 1 + (x0 // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr2 + (x5), None)
    tmp40 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (((-1) + x1) // 2))
    tmp10 = tl.math.min(28, 1 + (x1 // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (((-1) + x0) // 2)))
    tmp13 = tl.math.min(28, 1 + (x0 // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (((-1) + x1) // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (((-1) + x0) // 2))
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


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlniutptb5mtlgiuy6spnuqnbmc3ra52xxwmcorayy4oye7xeia.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/62/c62wm25ihqpfociuzcudro3wvkcifp3xvbwpxc4mfv7kmwwbxrxg.py
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
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
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (2007040 + (3136*x0) + (2408448*(r2 // 3136)) + (4816896*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4rxwkp444bssbn4dhipwtouyryxdadb6gkwncmofv65vaz7mv4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3jq36mt3n3domuacwiiupftclkbk6pgaqaknfdqfe5fsbihcrq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3pajsdstwa4pwsraez4ih53ibbgtckyjnodazipxllfkcum2db.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (2007040 + x4 + (2408448*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3ijtrs5atnz24ksnbqessbfmzrtyspsmznq4gamhsiekco2z6yy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (1605632 + (3136*x0) + (2408448*(r2 // 3136)) + (4816896*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coy2anykwgty4a3wesm3xalwoarhtlyqbbn5gy2ayxbvnhdr6oxj.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (1605632 + x4 + (2408448*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmx25qgnpowgsokv3n4uasa2edox6sgjng6x5qyk7lpvploo5lbq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (1204224 + (3136*x0) + (2408448*(r2 // 3136)) + (4816896*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cni6zrrsgtomiyjjffpdr2tvdkgcbxkssi42npnkht23q5b246qq.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (1204224 + x4 + (2408448*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czftvbtjaec6kx5vbbx4wveanu5n4lth2lgygrl7xnifnwdam5j5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (802816 + (3136*x0) + (2408448*(r2 // 3136)) + (4816896*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2zetrzgxuz6ff72s24j4u2vubcneqta7bbxc3xsb3yaplepcbe.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (802816 + x4 + (2408448*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7w7w2betyxmvjpznxs2q33f2r7a3ucuto6k7dbk6hdvvm3agqw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (401408 + (3136*x0) + (2408448*(r2 // 3136)) + (4816896*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujjzquuwlmsm4clruoe4uk4pige3tnvw2hmoi5iqkhio7cz22w2.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (401408 + x4 + (2408448*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3a5bsd4xpm3sw5k6ty374vjpxfpvmoov4l4yztodwxqlk55edim.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (2408448*(r2 // 3136)) + (4816896*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujban7gcmmpcbvzfwzipg37krk6ls36lsf7mafkojfhgv74oomx.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x4 + (2408448*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckuklyww5jggs3encv3lifl4jtyzfwacl2yedbotmqsj7ks6ztqk.py
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/er/cercisvld43w7isgzmeagbbtzxeitkxmifjnqieendbchq5r6vys.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/m5/cm53mozm6ptmh2bmowhfvugft6cvhoirvjbqvxaydavt66w43uqn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_82', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/bl/cblw3yt2g5cap2dxtc6ixu6y2ufrx5j3sp7xs4jchtz6kjixo4h6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, convolution_3, relu_3, convolution_4, relu_4, convolution_5, relu_5, convolution_6, relu_6, convolution_7, cat, convolution_8, relu_8, getitem, getitem_1, convolution_9, relu_9, convolution_10, relu_10, convolution_11, relu_11, convolution_12, relu_12, convolution_13, cat_1, convolution_14, relu_14, getitem_2, getitem_3, convolution_15, relu_15, convolution_16, relu_16, convolution_17, relu_17, convolution_18, relu_18, convolution_19, cat_2, convolution_20, relu_20, convolution_21, relu_21, convolution_22, relu_22, convolution_23, relu_23, convolution_24, relu_24, convolution_25, cat_3, convolution_26, relu_26, getitem_4, getitem_5, convolution_27, relu_27, convolution_28, relu_28, convolution_29, relu_29, convolution_30, relu_30, convolution_31, cat_4, convolution_32, relu_32, convolution_33, relu_33, convolution_34, relu_34, convolution_35, relu_35, convolution_36, relu_36, convolution_37, cat_5, convolution_38, clone, permute_1, le, le_1, le_7, le_13, le_19, le_25, le_31, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_25, (160, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (224, ), (1, ))
    assert_size_stride(primals_57, (224, ), (1, ))
    assert_size_stride(primals_59, (224, ), (1, ))
    assert_size_stride(primals_61, (224, ), (1, ))
    assert_size_stride(primals_63, (224, ), (1, ))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_67, (224, ), (1, ))
    assert_size_stride(primals_69, (224, ), (1, ))
    assert_size_stride(primals_71, (224, ), (1, ))
    assert_size_stride(primals_73, (224, ), (1, ))
    assert_size_stride(primals_75, (224, ), (1, ))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_79, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_80, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_81, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_83, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_84, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_85, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_86, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_87, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_88, (160, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_89, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_90, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_91, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_92, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_93, (512, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_94, (192, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_95, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_96, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_97, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_98, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_99, (768, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(primals_100, (192, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_101, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_102, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_103, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_104, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_105, (768, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(primals_106, (224, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_107, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_108, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_109, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_110, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_111, (1024, 1888, 1, 1), (1888, 1, 1, 1))
    assert_size_stride(primals_112, (224, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_113, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_114, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_115, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_116, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_117, (1024, 2144, 1, 1), (2144, 1, 1, 1))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, ), (1, ))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (160, ), (1, ))
    assert_size_stride(primals_139, (160, ), (1, ))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_141, (160, ), (1, ))
    assert_size_stride(primals_142, (160, ), (1, ))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_144, (160, ), (1, ))
    assert_size_stride(primals_145, (160, ), (1, ))
    assert_size_stride(primals_146, (160, ), (1, ))
    assert_size_stride(primals_147, (160, ), (1, ))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (192, ), (1, ))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (192, ), (1, ))
    assert_size_stride(primals_155, (192, ), (1, ))
    assert_size_stride(primals_156, (192, ), (1, ))
    assert_size_stride(primals_157, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_159, (192, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (192, ), (1, ))
    assert_size_stride(primals_163, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_165, (192, ), (1, ))
    assert_size_stride(primals_166, (192, ), (1, ))
    assert_size_stride(primals_167, (192, ), (1, ))
    assert_size_stride(primals_168, (192, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_170, (192, ), (1, ))
    assert_size_stride(primals_171, (192, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (224, ), (1, ))
    assert_size_stride(primals_175, (224, ), (1, ))
    assert_size_stride(primals_176, (224, ), (1, ))
    assert_size_stride(primals_177, (224, ), (1, ))
    assert_size_stride(primals_178, (224, ), (1, ))
    assert_size_stride(primals_179, (224, ), (1, ))
    assert_size_stride(primals_180, (224, ), (1, ))
    assert_size_stride(primals_181, (224, ), (1, ))
    assert_size_stride(primals_182, (224, ), (1, ))
    assert_size_stride(primals_183, (224, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (224, ), (1, ))
    assert_size_stride(primals_187, (224, ), (1, ))
    assert_size_stride(primals_188, (224, ), (1, ))
    assert_size_stride(primals_189, (224, ), (1, ))
    assert_size_stride(primals_190, (224, ), (1, ))
    assert_size_stride(primals_191, (224, ), (1, ))
    assert_size_stride(primals_192, (224, ), (1, ))
    assert_size_stride(primals_193, (224, ), (1, ))
    assert_size_stride(primals_194, (224, ), (1, ))
    assert_size_stride(primals_195, (224, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (4, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(relu, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(convolution_1, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(relu_1, (4, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(convolution_2, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_2, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_3, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_3, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_4, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_4, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_5, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_5, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_6, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(relu_6, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_7, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(cat, (4, 768, 56, 56), (2408448, 3136, 56, 1))
    assert_size_stride(convolution_8, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(relu_8, (4, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(getitem, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(getitem_1, (4, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_9, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(relu_9, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(convolution_10, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(relu_10, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(convolution_11, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(relu_11, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(convolution_12, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(relu_12, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(convolution_13, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(cat_1, (4, 1056, 28, 28), (827904, 784, 28, 1))
    assert_size_stride(convolution_14, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(relu_14, (4, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(getitem_2, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(getitem_3, (4, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_15, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_15, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_16, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_16, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_17, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_17, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_18, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_18, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_19, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(cat_2, (4, 1472, 14, 14), (288512, 196, 14, 1))
    assert_size_stride(convolution_20, (4, 768, 14, 14), (150528, 196, 14, 1))
    assert_size_stride(relu_20, (4, 768, 14, 14), (150528, 196, 14, 1))
    assert_size_stride(convolution_21, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_21, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_22, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_22, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_23, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_23, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_24, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(relu_24, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(convolution_25, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(cat_3, (4, 1728, 14, 14), (338688, 196, 14, 1))
    assert_size_stride(convolution_26, (4, 768, 14, 14), (150528, 196, 14, 1))
    assert_size_stride(relu_26, (4, 768, 14, 14), (150528, 196, 14, 1))
    assert_size_stride(getitem_4, (4, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(getitem_5, (4, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_27, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_27, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_28, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_28, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_29, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_29, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_30, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_30, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_31, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(cat_4, (4, 1888, 7, 7), (92512, 49, 7, 1))
    assert_size_stride(convolution_32, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(relu_32, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_33, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_33, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_34, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_34, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_35, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_35, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_36, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(relu_36, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(convolution_37, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(cat_5, (4, 2144, 7, 7), (105056, 49, 7, 1))
    assert_size_stride(convolution_38, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(clone, (4, 1024), (1024, 1))
    assert_size_stride(permute_1, (1000, 1024), (1024, 1))
    assert_size_stride(le, (4, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(le_1, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(le_7, (4, 224, 7, 7), (10976, 49, 7, 1))
    assert_size_stride(le_13, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(le_19, (4, 192, 14, 14), (37632, 196, 14, 1))
    assert_size_stride(le_25, (4, 160, 28, 28), (125440, 784, 28, 1))
    assert_size_stride(le_31, (4, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf4 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(buf5, le, buf0, convolution_38, primals_196, primals_197, buf3, 1024, 196, grid=grid(1024), stream=stream0)
        del convolution_38
        del primals_196
        buf6 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2.run(le, buf0, primals_197, primals_77, buf6, 200704, grid=grid(200704), stream=stream0)
        del buf0
        del le
        del primals_197
        del primals_77
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, cat_5, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf6
        del cat_5
        del primals_117
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = empty((224, ), device='cuda', dtype=torch.float32)
        buf11 = empty((224, ), device='cuda', dtype=torch.float32)
        buf12 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(buf12, le_1, buf8, convolution_37, primals_194, primals_195, buf10, 224, 196, grid=grid(224), stream=stream0)
        del convolution_37
        del primals_194
        buf13 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(le_1, buf8, primals_195, primals_75, buf13, 43904, grid=grid(43904), stream=stream0)
        del le_1
        del primals_195
        del primals_75
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf14 = aten.convolution_backward(buf13, relu_36, primals_116, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf13
        del primals_116
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = empty((224, ), device='cuda', dtype=torch.float32)
        buf18 = empty((224, ), device='cuda', dtype=torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_5.run(buf19, relu_36, buf8, buf15, convolution_36, primals_192, primals_193, buf17, 224, 196, grid=grid(224), stream=stream0)
        del convolution_36
        del primals_192
        buf20 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(buf20, relu_36, buf8, primals_193, primals_73, 43904, grid=grid(43904), stream=stream0)
        del primals_193
        del primals_73
        del relu_36
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf21 = aten.convolution_backward(buf20, relu_35, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf20
        del primals_115
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = empty((224, ), device='cuda', dtype=torch.float32)
        buf25 = empty((224, ), device='cuda', dtype=torch.float32)
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(buf26, relu_35, buf8, buf22, convolution_35, primals_190, primals_191, buf24, 224, 196, grid=grid(224), stream=stream0)
        del convolution_35
        del primals_190
        buf27 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(buf27, relu_35, buf8, primals_191, primals_71, 43904, grid=grid(43904), stream=stream0)
        del primals_191
        del primals_71
        del relu_35
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf28 = aten.convolution_backward(buf27, relu_34, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf27
        del primals_114
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = empty((224, ), device='cuda', dtype=torch.float32)
        buf32 = empty((224, ), device='cuda', dtype=torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(buf33, relu_34, buf8, buf29, convolution_34, primals_188, primals_189, buf31, 224, 196, grid=grid(224), stream=stream0)
        del convolution_34
        del primals_188
        buf34 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf34, relu_34, buf8, primals_189, primals_69, 43904, grid=grid(43904), stream=stream0)
        del primals_189
        del primals_69
        del relu_34
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf35 = aten.convolution_backward(buf34, relu_33, primals_113, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf34
        del primals_113
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((224, ), device='cuda', dtype=torch.float32)
        buf39 = empty((224, ), device='cuda', dtype=torch.float32)
        buf40 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_11.run(buf40, relu_33, buf8, buf36, convolution_33, primals_186, primals_187, buf38, 224, 196, grid=grid(224), stream=stream0)
        del convolution_33
        del primals_186
        buf41 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(buf41, relu_33, buf8, primals_187, primals_67, 43904, grid=grid(43904), stream=stream0)
        del primals_187
        del primals_67
        del relu_33
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf42 = aten.convolution_backward(buf41, relu_32, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_112
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf45 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf46 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_13.run(buf47, relu_32, buf8, buf43, convolution_32, primals_184, primals_185, buf45, 1024, 196, grid=grid(1024), stream=stream0)
        del convolution_32
        del primals_184
        buf48 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf48, relu_32, buf8, primals_185, primals_65, 200704, grid=grid(200704), stream=stream0)
        del buf8
        del primals_185
        del primals_65
        del relu_32
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf49 = aten.convolution_backward(buf48, cat_4, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf48
        del cat_4
        del primals_111
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = empty((224, ), device='cuda', dtype=torch.float32)
        buf53 = empty((224, ), device='cuda', dtype=torch.float32)
        buf54 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_15.run(buf54, le_7, buf50, convolution_31, primals_182, primals_183, buf52, 224, 196, grid=grid(224), stream=stream0)
        del convolution_31
        del primals_182
        buf55 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16.run(le_7, buf50, primals_183, primals_63, buf55, 43904, grid=grid(43904), stream=stream0)
        del le_7
        del primals_183
        del primals_63
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf56 = aten.convolution_backward(buf55, relu_30, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf55
        del primals_110
        buf57 = buf56[0]
        buf58 = buf56[1]
        del buf56
        buf59 = empty((224, ), device='cuda', dtype=torch.float32)
        buf60 = empty((224, ), device='cuda', dtype=torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_17.run(buf61, relu_30, buf50, buf57, convolution_30, primals_180, primals_181, buf59, 224, 196, grid=grid(224), stream=stream0)
        del convolution_30
        del primals_180
        buf62 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(buf62, relu_30, buf50, primals_181, primals_61, 43904, grid=grid(43904), stream=stream0)
        del primals_181
        del primals_61
        del relu_30
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf63 = aten.convolution_backward(buf62, relu_29, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf62
        del primals_109
        buf64 = buf63[0]
        buf65 = buf63[1]
        del buf63
        buf66 = empty((224, ), device='cuda', dtype=torch.float32)
        buf67 = empty((224, ), device='cuda', dtype=torch.float32)
        buf68 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_19.run(buf68, relu_29, buf50, buf64, convolution_29, primals_178, primals_179, buf66, 224, 196, grid=grid(224), stream=stream0)
        del convolution_29
        del primals_178
        buf69 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20.run(buf69, relu_29, buf50, primals_179, primals_59, 43904, grid=grid(43904), stream=stream0)
        del primals_179
        del primals_59
        del relu_29
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf70 = aten.convolution_backward(buf69, relu_28, primals_108, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf69
        del primals_108
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = empty((224, ), device='cuda', dtype=torch.float32)
        buf74 = empty((224, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_21.run(buf75, relu_28, buf50, buf71, convolution_28, primals_176, primals_177, buf73, 224, 196, grid=grid(224), stream=stream0)
        del convolution_28
        del primals_176
        buf76 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22.run(buf76, relu_28, buf50, primals_177, primals_57, 43904, grid=grid(43904), stream=stream0)
        del primals_177
        del primals_57
        del relu_28
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf77 = aten.convolution_backward(buf76, relu_27, primals_107, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf76
        del primals_107
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = empty((224, ), device='cuda', dtype=torch.float32)
        buf81 = empty((224, ), device='cuda', dtype=torch.float32)
        buf82 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf82, relu_27, buf50, buf78, convolution_27, primals_174, primals_175, buf80, 224, 196, grid=grid(224), stream=stream0)
        del convolution_27
        del primals_174
        buf83 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf83, relu_27, buf50, primals_175, primals_55, 43904, grid=grid(43904), stream=stream0)
        del primals_175
        del primals_55
        del relu_27
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf84 = aten.convolution_backward(buf83, getitem_4, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf83
        del getitem_4
        del primals_106
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf87 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_25.run(buf87, buf50, 150528, grid=grid(150528), stream=stream0)
        del buf50
        buf88 = empty((4, 768, 14, 14), device='cuda', dtype=torch.float32)
        buf92 = empty((4, 768, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_26.run(getitem_5, buf87, relu_26, primals_173, primals_53, buf88, buf92, 602112, grid=grid(602112), stream=stream0)
        del getitem_5
        del primals_53
        buf89 = empty((768, ), device='cuda', dtype=torch.float32)
        buf90 = empty((768, ), device='cuda', dtype=torch.float32)
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_27.run(buf91, relu_26, buf88, convolution_26, primals_172, primals_173, buf89, 768, 784, grid=grid(768), stream=stream0)
        del buf88
        del convolution_26
        del primals_172
        del primals_173
        del relu_26
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf93 = aten.convolution_backward(buf92, cat_3, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf92
        del cat_3
        del primals_105
        buf94 = buf93[0]
        buf95 = buf93[1]
        del buf93
        buf96 = empty((192, ), device='cuda', dtype=torch.float32)
        buf97 = empty((192, ), device='cuda', dtype=torch.float32)
        buf98 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf98, le_13, buf94, convolution_25, primals_170, primals_171, buf96, 192, 784, grid=grid(192), stream=stream0)
        del convolution_25
        del primals_170
        buf99 = reinterpret_tensor(buf87, (4, 192, 14, 14), (37632, 196, 14, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(le_13, buf94, primals_171, primals_51, buf99, 150528, grid=grid(150528), stream=stream0)
        del le_13
        del primals_171
        del primals_51
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf100 = aten.convolution_backward(buf99, relu_24, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf99
        del primals_104
        buf101 = buf100[0]
        buf102 = buf100[1]
        del buf100
        buf103 = empty((192, ), device='cuda', dtype=torch.float32)
        buf104 = empty((192, ), device='cuda', dtype=torch.float32)
        buf105 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_30.run(buf105, relu_24, buf94, buf101, convolution_24, primals_168, primals_169, buf103, 192, 784, grid=grid(192), stream=stream0)
        del convolution_24
        del primals_168
        buf106 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf106, relu_24, buf94, primals_169, primals_49, 150528, grid=grid(150528), stream=stream0)
        del primals_169
        del primals_49
        del relu_24
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf107 = aten.convolution_backward(buf106, relu_23, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf106
        del primals_103
        buf108 = buf107[0]
        buf109 = buf107[1]
        del buf107
        buf110 = empty((192, ), device='cuda', dtype=torch.float32)
        buf111 = empty((192, ), device='cuda', dtype=torch.float32)
        buf112 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_32.run(buf112, relu_23, buf94, buf108, convolution_23, primals_166, primals_167, buf110, 192, 784, grid=grid(192), stream=stream0)
        del convolution_23
        del primals_166
        buf113 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_33.run(buf113, relu_23, buf94, primals_167, primals_47, 150528, grid=grid(150528), stream=stream0)
        del primals_167
        del primals_47
        del relu_23
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf114 = aten.convolution_backward(buf113, relu_22, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf113
        del primals_102
        buf115 = buf114[0]
        buf116 = buf114[1]
        del buf114
        buf117 = empty((192, ), device='cuda', dtype=torch.float32)
        buf118 = empty((192, ), device='cuda', dtype=torch.float32)
        buf119 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_34.run(buf119, relu_22, buf94, buf115, convolution_22, primals_164, primals_165, buf117, 192, 784, grid=grid(192), stream=stream0)
        del convolution_22
        del primals_164
        buf120 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_35.run(buf120, relu_22, buf94, primals_165, primals_45, 150528, grid=grid(150528), stream=stream0)
        del primals_165
        del primals_45
        del relu_22
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf121 = aten.convolution_backward(buf120, relu_21, primals_101, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf120
        del primals_101
        buf122 = buf121[0]
        buf123 = buf121[1]
        del buf121
        buf124 = empty((192, ), device='cuda', dtype=torch.float32)
        buf125 = empty((192, ), device='cuda', dtype=torch.float32)
        buf126 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_36.run(buf126, relu_21, buf94, buf122, convolution_21, primals_162, primals_163, buf124, 192, 784, grid=grid(192), stream=stream0)
        del convolution_21
        del primals_162
        buf127 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37.run(buf127, relu_21, buf94, primals_163, primals_43, 150528, grid=grid(150528), stream=stream0)
        del primals_163
        del primals_43
        del relu_21
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf128 = aten.convolution_backward(buf127, relu_20, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf129 = buf128[0]
        buf130 = buf128[1]
        del buf128
        buf131 = empty((768, ), device='cuda', dtype=torch.float32)
        buf132 = empty((768, ), device='cuda', dtype=torch.float32)
        buf133 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_38.run(buf133, relu_20, buf94, buf129, convolution_20, primals_160, primals_161, buf131, 768, 784, grid=grid(768), stream=stream0)
        del convolution_20
        del primals_160
        buf134 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(buf134, relu_20, buf94, primals_161, primals_41, 602112, grid=grid(602112), stream=stream0)
        del buf94
        del primals_161
        del primals_41
        del relu_20
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf135 = aten.convolution_backward(buf134, cat_2, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf134
        del cat_2
        del primals_99
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf138 = empty((192, ), device='cuda', dtype=torch.float32)
        buf139 = empty((192, ), device='cuda', dtype=torch.float32)
        buf140 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf140, le_19, buf136, convolution_19, primals_158, primals_159, buf138, 192, 784, grid=grid(192), stream=stream0)
        del convolution_19
        del primals_158
        buf141 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41.run(le_19, buf136, primals_159, primals_39, buf141, 150528, grid=grid(150528), stream=stream0)
        del le_19
        del primals_159
        del primals_39
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf142 = aten.convolution_backward(buf141, relu_18, primals_98, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf141
        del primals_98
        buf143 = buf142[0]
        buf144 = buf142[1]
        del buf142
        buf145 = empty((192, ), device='cuda', dtype=torch.float32)
        buf146 = empty((192, ), device='cuda', dtype=torch.float32)
        buf147 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_42.run(buf147, relu_18, buf136, buf143, convolution_18, primals_156, primals_157, buf145, 192, 784, grid=grid(192), stream=stream0)
        del convolution_18
        del primals_156
        buf148 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf148, relu_18, buf136, primals_157, primals_37, 150528, grid=grid(150528), stream=stream0)
        del primals_157
        del primals_37
        del relu_18
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf149 = aten.convolution_backward(buf148, relu_17, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf148
        del primals_97
        buf150 = buf149[0]
        buf151 = buf149[1]
        del buf149
        buf152 = empty((192, ), device='cuda', dtype=torch.float32)
        buf153 = empty((192, ), device='cuda', dtype=torch.float32)
        buf154 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf154, relu_17, buf136, buf150, convolution_17, primals_154, primals_155, buf152, 192, 784, grid=grid(192), stream=stream0)
        del convolution_17
        del primals_154
        buf155 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf155, relu_17, buf136, primals_155, primals_35, 150528, grid=grid(150528), stream=stream0)
        del primals_155
        del primals_35
        del relu_17
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf156 = aten.convolution_backward(buf155, relu_16, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf155
        del primals_96
        buf157 = buf156[0]
        buf158 = buf156[1]
        del buf156
        buf159 = empty((192, ), device='cuda', dtype=torch.float32)
        buf160 = empty((192, ), device='cuda', dtype=torch.float32)
        buf161 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf161, relu_16, buf136, buf157, convolution_16, primals_152, primals_153, buf159, 192, 784, grid=grid(192), stream=stream0)
        del convolution_16
        del primals_152
        buf162 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_47.run(buf162, relu_16, buf136, primals_153, primals_33, 150528, grid=grid(150528), stream=stream0)
        del primals_153
        del primals_33
        del relu_16
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf163 = aten.convolution_backward(buf162, relu_15, primals_95, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf162
        del primals_95
        buf164 = buf163[0]
        buf165 = buf163[1]
        del buf163
        buf166 = empty((192, ), device='cuda', dtype=torch.float32)
        buf167 = empty((192, ), device='cuda', dtype=torch.float32)
        buf168 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_48.run(buf168, relu_15, buf136, buf164, convolution_15, primals_150, primals_151, buf166, 192, 784, grid=grid(192), stream=stream0)
        del convolution_15
        del primals_150
        buf169 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(buf169, relu_15, buf136, primals_151, primals_31, 150528, grid=grid(150528), stream=stream0)
        del primals_151
        del primals_31
        del relu_15
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf170 = aten.convolution_backward(buf169, getitem_2, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf169
        del getitem_2
        del primals_94
        buf171 = buf170[0]
        buf172 = buf170[1]
        del buf170
        buf173 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf173, buf136, 401408, grid=grid(401408), stream=stream0)
        del buf136
        buf174 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        buf178 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_51.run(getitem_3, buf173, relu_14, primals_149, primals_29, buf174, buf178, 1605632, grid=grid(1605632), stream=stream0)
        del buf173
        del getitem_3
        del primals_29
        buf175 = empty((512, ), device='cuda', dtype=torch.float32)
        buf176 = empty((512, ), device='cuda', dtype=torch.float32)
        buf177 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_52.run(buf177, relu_14, buf174, convolution_14, primals_148, primals_149, buf175, 512, 3136, grid=grid(512), stream=stream0)
        del buf174
        del convolution_14
        del primals_148
        del primals_149
        del relu_14
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf179 = aten.convolution_backward(buf178, cat_1, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_93
        buf180 = buf179[0]
        buf181 = buf179[1]
        del buf179
        buf182 = empty((160, ), device='cuda', dtype=torch.float32)
        buf183 = empty((160, ), device='cuda', dtype=torch.float32)
        buf184 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(buf184, le_25, buf180, convolution_13, primals_146, primals_147, buf182, 160, 3136, grid=grid(160), stream=stream0)
        del convolution_13
        del primals_146
        buf185 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54.run(le_25, buf180, primals_147, primals_27, buf185, 501760, grid=grid(501760), stream=stream0)
        del le_25
        del primals_147
        del primals_27
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf186 = aten.convolution_backward(buf185, relu_12, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf185
        del primals_92
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf189 = empty((160, ), device='cuda', dtype=torch.float32)
        buf190 = empty((160, ), device='cuda', dtype=torch.float32)
        buf191 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(buf191, relu_12, buf180, buf187, convolution_12, primals_144, primals_145, buf189, 160, 3136, grid=grid(160), stream=stream0)
        del convolution_12
        del primals_144
        buf192 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf192, relu_12, buf180, primals_145, primals_25, 501760, grid=grid(501760), stream=stream0)
        del primals_145
        del primals_25
        del relu_12
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf193 = aten.convolution_backward(buf192, relu_11, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf192
        del primals_91
        buf194 = buf193[0]
        buf195 = buf193[1]
        del buf193
        buf196 = empty((160, ), device='cuda', dtype=torch.float32)
        buf197 = empty((160, ), device='cuda', dtype=torch.float32)
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_57.run(buf198, relu_11, buf180, buf194, convolution_11, primals_142, primals_143, buf196, 160, 3136, grid=grid(160), stream=stream0)
        del convolution_11
        del primals_142
        buf199 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_58.run(buf199, relu_11, buf180, primals_143, primals_23, 501760, grid=grid(501760), stream=stream0)
        del primals_143
        del primals_23
        del relu_11
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf200 = aten.convolution_backward(buf199, relu_10, primals_90, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf199
        del primals_90
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = empty((160, ), device='cuda', dtype=torch.float32)
        buf204 = empty((160, ), device='cuda', dtype=torch.float32)
        buf205 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_59.run(buf205, relu_10, buf180, buf201, convolution_10, primals_140, primals_141, buf203, 160, 3136, grid=grid(160), stream=stream0)
        del convolution_10
        del primals_140
        buf206 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf206, relu_10, buf180, primals_141, primals_21, 501760, grid=grid(501760), stream=stream0)
        del primals_141
        del primals_21
        del relu_10
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf207 = aten.convolution_backward(buf206, relu_9, primals_89, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf206
        del primals_89
        buf208 = buf207[0]
        buf209 = buf207[1]
        del buf207
        buf210 = empty((160, ), device='cuda', dtype=torch.float32)
        buf211 = empty((160, ), device='cuda', dtype=torch.float32)
        buf212 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_61.run(buf212, relu_9, buf180, buf208, convolution_9, primals_138, primals_139, buf210, 160, 3136, grid=grid(160), stream=stream0)
        del convolution_9
        del primals_138
        buf213 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_62.run(buf213, relu_9, buf180, primals_139, primals_19, 501760, grid=grid(501760), stream=stream0)
        del primals_139
        del primals_19
        del relu_9
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf214 = aten.convolution_backward(buf213, getitem, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf213
        del getitem
        del primals_88
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf217 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_63.run(buf217, buf180, 802816, grid=grid(802816), stream=stream0)
        del buf180
        buf218 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        buf222 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.max_pool2d_with_indices_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_threshold_backward_64.run(getitem_1, buf217, relu_8, primals_137, primals_17, buf218, buf222, 3211264, grid=grid(3211264), stream=stream0)
        del buf217
        del getitem_1
        del primals_17
        buf219 = empty((256, ), device='cuda', dtype=torch.float32)
        buf220 = empty((256, ), device='cuda', dtype=torch.float32)
        buf221 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf221, relu_8, buf218, convolution_8, primals_136, primals_137, buf219, 256, 12544, grid=grid(256), stream=stream0)
        del buf218
        del convolution_8
        del primals_136
        del primals_137
        del relu_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf223 = aten.convolution_backward(buf222, cat, primals_87, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf222
        del cat
        del primals_87
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        buf226 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        buf228 = empty_strided((128, 2), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(le_31, buf224, convolution_7, primals_134, buf226, buf228, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_7
        del primals_134
        buf227 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf226, buf227, 128, 2, grid=grid(128), stream=stream0)
        buf229 = empty((128, ), device='cuda', dtype=torch.float32)
        buf230 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_68.run(buf230, buf228, primals_135, 128, 2, grid=grid(128), stream=stream0)
        buf231 = reinterpret_tensor(buf178, (4, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69.run(le_31, buf224, primals_135, primals_15, buf231, 1605632, grid=grid(1605632), stream=stream0)
        del le_31
        del primals_135
        del primals_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf232 = aten.convolution_backward(buf231, relu_6, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf231
        del primals_86
        buf233 = buf232[0]
        buf234 = buf232[1]
        del buf232
        buf235 = buf228; del buf228  # reuse
        buf237 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_70.run(relu_6, buf224, buf233, convolution_6, primals_132, buf235, buf237, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_6
        del primals_132
        buf236 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf235, buf236, 128, 2, grid=grid(128), stream=stream0)
        buf238 = empty((128, ), device='cuda', dtype=torch.float32)
        buf239 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_68.run(buf239, buf237, primals_133, 128, 2, grid=grid(128), stream=stream0)
        buf240 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_71.run(buf240, relu_6, buf224, primals_133, primals_13, 1605632, grid=grid(1605632), stream=stream0)
        del primals_13
        del primals_133
        del relu_6
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf241 = aten.convolution_backward(buf240, relu_5, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf240
        del primals_85
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        buf244 = buf237; del buf237  # reuse
        buf246 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_72.run(relu_5, buf224, buf242, convolution_5, primals_130, buf244, buf246, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_5
        del primals_130
        buf245 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf244, buf245, 128, 2, grid=grid(128), stream=stream0)
        buf247 = empty((128, ), device='cuda', dtype=torch.float32)
        buf248 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_68.run(buf248, buf246, primals_131, 128, 2, grid=grid(128), stream=stream0)
        buf249 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf249, relu_5, buf224, primals_131, primals_11, 1605632, grid=grid(1605632), stream=stream0)
        del primals_11
        del primals_131
        del relu_5
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf250 = aten.convolution_backward(buf249, relu_4, primals_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf249
        del primals_84
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf253 = buf246; del buf246  # reuse
        buf255 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_74.run(relu_4, buf224, buf251, convolution_4, primals_128, buf253, buf255, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_4
        del primals_128
        buf254 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf253, buf254, 128, 2, grid=grid(128), stream=stream0)
        buf256 = empty((128, ), device='cuda', dtype=torch.float32)
        buf257 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_68.run(buf257, buf255, primals_129, 128, 2, grid=grid(128), stream=stream0)
        buf258 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(buf258, relu_4, buf224, primals_129, primals_9, 1605632, grid=grid(1605632), stream=stream0)
        del primals_129
        del primals_9
        del relu_4
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf259 = aten.convolution_backward(buf258, relu_3, primals_83, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf258
        del primals_83
        buf260 = buf259[0]
        buf261 = buf259[1]
        del buf259
        buf262 = buf255; del buf255  # reuse
        buf264 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_76.run(relu_3, buf224, buf260, convolution_3, primals_126, buf262, buf264, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_3
        del primals_126
        buf263 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf262, buf263, 128, 2, grid=grid(128), stream=stream0)
        buf265 = empty((128, ), device='cuda', dtype=torch.float32)
        buf266 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_68.run(buf266, buf264, primals_127, 128, 2, grid=grid(128), stream=stream0)
        buf267 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_77.run(buf267, relu_3, buf224, primals_127, primals_7, 1605632, grid=grid(1605632), stream=stream0)
        del primals_127
        del primals_7
        del relu_3
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf268 = aten.convolution_backward(buf267, relu_2, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf267
        del primals_82
        buf269 = buf268[0]
        buf270 = buf268[1]
        del buf268
        buf271 = buf264; del buf264  # reuse
        buf273 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_78.run(relu_2, buf224, buf269, convolution_2, primals_124, buf271, buf273, 256, 6272, grid=grid(256), stream=stream0)
        del convolution_2
        del primals_124
        buf272 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf271, buf272, 128, 2, grid=grid(128), stream=stream0)
        del buf271
        buf274 = empty((128, ), device='cuda', dtype=torch.float32)
        buf275 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_68.run(buf275, buf273, primals_125, 128, 2, grid=grid(128), stream=stream0)
        del buf273
        buf276 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_79.run(buf276, relu_2, buf224, primals_125, primals_5, 1605632, grid=grid(1605632), stream=stream0)
        del buf224
        del primals_125
        del primals_5
        del relu_2
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf277 = aten.convolution_backward(buf276, relu_1, primals_81, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf276
        del primals_81
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = empty((64, 7), device='cuda', dtype=torch.float32)
        buf282 = empty((64, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_1, buf278, convolution_1, primals_122, buf280, buf282, 448, 7168, grid=grid(448), stream=stream0)
        del convolution_1
        del primals_122
        buf281 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf280, buf281, 64, 7, grid=grid(64), stream=stream0)
        buf283 = empty((64, ), device='cuda', dtype=torch.float32)
        buf284 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_82.run(buf284, buf282, primals_123, 64, 7, grid=grid(64), stream=stream0)
        buf285 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf285, relu_1, primals_123, primals_3, 3211264, grid=grid(3211264), stream=stream0)
        del primals_123
        del primals_3
        del relu_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf286 = aten.convolution_backward(buf285, relu, primals_80, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf285
        del primals_80
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        buf289 = buf282; del buf282  # reuse
        buf291 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu, buf287, convolution, primals_120, buf289, buf291, 448, 7168, grid=grid(448), stream=stream0)
        del convolution
        del primals_120
        buf290 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf289, buf290, 64, 7, grid=grid(64), stream=stream0)
        del buf289
        buf292 = empty((64, ), device='cuda', dtype=torch.float32)
        buf293 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_82.run(buf293, buf291, primals_121, 64, 7, grid=grid(64), stream=stream0)
        del buf291
        buf294 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf294, relu, primals_121, primals_1, 3211264, grid=grid(3211264), stream=stream0)
        del primals_1
        del primals_121
        del relu
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf295 = aten.convolution_backward(buf294, primals_198, primals_79, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf294
        del primals_198
        del primals_79
        buf296 = buf295[1]
        return (buf293, buf290, buf284, buf281, buf275, buf272, buf266, buf263, buf257, buf254, buf248, buf245, buf239, buf236, buf230, buf227, buf221, buf219, buf212, buf210, buf205, buf203, buf198, buf196, buf191, buf189, buf184, buf182, buf177, buf175, buf168, buf166, buf161, buf159, buf154, buf152, buf147, buf145, buf140, buf138, buf133, buf131, buf126, buf124, buf119, buf117, buf112, buf110, buf105, buf103, buf98, buf96, buf91, buf89, buf82, buf80, buf75, buf73, buf68, buf66, buf61, buf59, buf54, buf52, buf47, buf45, buf40, buf38, buf33, buf31, buf26, buf24, buf19, buf17, buf12, buf10, buf5, buf3, buf296, buf288, buf279, buf270, buf261, buf252, buf243, buf234, buf225, buf216, buf209, buf202, buf195, buf188, buf181, buf172, buf165, buf158, buf151, buf144, buf137, buf130, buf123, buf116, buf109, buf102, buf95, buf86, buf79, buf72, buf65, buf58, buf51, buf44, buf37, buf30, buf23, buf16, buf9, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((160, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((192, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((224, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, 1888, 1, 1), (1888, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((224, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, 2144, 1, 1), (2144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((4, 768, 56, 56), (2408448, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.int64)
    convolution_9 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((4, 1056, 28, 28), (827904, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((4, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((4, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.int64)
    convolution_15 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((4, 1472, 14, 14), (288512, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 768, 14, 14), (150528, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((4, 768, 14, 14), (150528, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((4, 1728, 14, 14), (338688, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 768, 14, 14), (150528, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((4, 768, 14, 14), (150528, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((4, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((4, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.int64)
    convolution_27 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((4, 1888, 7, 7), (92512, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((4, 2144, 7, 7), (105056, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    le_1 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    le_7 = rand_strided((4, 224, 7, 7), (10976, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    le_13 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    le_19 = rand_strided((4, 192, 14, 14), (37632, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    le_25 = rand_strided((4, 160, 28, 28), (125440, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    le_31 = rand_strided((4, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, convolution_3, relu_3, convolution_4, relu_4, convolution_5, relu_5, convolution_6, relu_6, convolution_7, cat, convolution_8, relu_8, getitem, getitem_1, convolution_9, relu_9, convolution_10, relu_10, convolution_11, relu_11, convolution_12, relu_12, convolution_13, cat_1, convolution_14, relu_14, getitem_2, getitem_3, convolution_15, relu_15, convolution_16, relu_16, convolution_17, relu_17, convolution_18, relu_18, convolution_19, cat_2, convolution_20, relu_20, convolution_21, relu_21, convolution_22, relu_22, convolution_23, relu_23, convolution_24, relu_24, convolution_25, cat_3, convolution_26, relu_26, getitem_4, getitem_5, convolution_27, relu_27, convolution_28, relu_28, convolution_29, relu_29, convolution_30, relu_30, convolution_31, cat_4, convolution_32, relu_32, convolution_33, relu_33, convolution_34, relu_34, convolution_35, relu_35, convolution_36, relu_36, convolution_37, cat_5, convolution_38, clone, permute_1, le, le_1, le_7, le_13, le_19, le_25, le_31, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_vovnet', benchmark_compiled_module)
