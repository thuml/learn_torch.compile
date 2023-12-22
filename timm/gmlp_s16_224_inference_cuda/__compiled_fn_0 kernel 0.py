
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


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2jiarujvy6azncpmon5w3egryo5lmdw65q7kmj7peztqriucal.py
# Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm => clone, var_mean
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 2
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/yh/cyh6ohyv2qzrx7lykwnedlbjz3dnwz3u3zm4ebtmvmdre2ul4hvx.py
# Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm => clone, var_mean
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwsdaemnurdkthlnwmljkgvmxtvln56ifry6rqhtgdwsc4e5shr.py
# Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3klvrw5inx73d2o5kdhgbhswpqfivwdwsfscghf37xmyc7q6ugr.py
# Source Nodes: [v_1], Original ATen: [aten.native_layer_norm]
# v_1 => clone_2, var_mean_1
triton_per_fused_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + r1 + (1536*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (768 + r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iy/ciy3dgh2uiwvo3bqsitob2bshsavozgukffhti7egmwyxs7kfsqn.py
# Source Nodes: [v_2], Original ATen: [aten.clone]
# v_2 => clone_3
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (1536*x2) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 768.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp19 = tmp12 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxableo4m7ubpoisp3ypfi4zmeoy4jdfnqampnze4c57vv6py3ln.py
# Source Nodes: [x_7], Original ATen: [aten.mul]
# x_7 => mul_7
triton_poi_fused_mul_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a3kwjkghx5ited3skaulnsviilmx7xhqejldr2d54jy74vhsfq.py
# Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm => clone_5, var_mean_2
# x_11 => add_6
triton_red_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 196
    x2 = (xindex // 392)
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (50176*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp9, xmask)
    tl.store(out_ptr2 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck23ksadhnvpf54lokwzdoiclosokaguylvqp4s5taxuzeitgtdv.py
# Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm => clone_5, var_mean_2
# x_11 => add_6
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfi7a7clxlcwrf2fpifgwnyhvbrq2zjo3rxkxufujcarky6idsl.py
# Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm => add_7, add_8, clone_5, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
# x_11 => add_6
triton_poi_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 256.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3xzthnfh4nxa5vl6sxquscsztu6vqscgzjxauswvle5sbpomnu.py
# Source Nodes: [getattr_l__mod___blocks___2___norm, x_11, x_19], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___2___norm => add_14, add_15, clone_10, mul_16, mul_17, rsqrt_4, sub_4, var_mean_4
# x_11 => add_6
# x_19 => add_13
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (50176*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 256, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 256.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5ilc2ae7eby47iuxa2powcdvnae5tiwhilny7x75ywd5rq42qmo.py
# Source Nodes: [getattr_l__mod___blocks___3___norm, x_27], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___3___norm => add_21, add_22, clone_15, mul_24, mul_25, rsqrt_6, sub_6, var_mean_6
# x_27 => add_20
triton_per_fused_add_native_layer_norm_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 256.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwccqcjdni2stoow2m5zjkot6mjximk5jpowmtaadlbtddrwv5fn.py
# Source Nodes: [getattr_l__mod___blocks___4___norm, x_27, x_35], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___4___norm => add_28, add_29, clone_20, mul_32, mul_33, rsqrt_8, sub_8, var_mean_8
# x_27 => add_20
# x_35 => add_27
triton_per_fused_add_native_layer_norm_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 256, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 256.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmkla3v4ugukx3nbakyuevngfgnpkyapcnediu3cydbstxclnaa.py
# Source Nodes: [x_235, x_244, x_246], Original ATen: [aten.add, aten.native_layer_norm]
# x_235 => add_202
# x_244 => add_209
# x_246 => clone_150, var_mean_60
triton_per_fused_add_native_layer_norm_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 256, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbppuydfr3onpskt3mktex6qsh65gfi4hcitklboemeefgzkc4b.py
# Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
# x_246 => add_210, add_211, mul_240, mul_241, rsqrt_60, sub_60, var_mean_60
# x_247 => mean
triton_red_fused_mean_native_layer_norm_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_layer_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 256.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-06
        tmp7 = tmp5 + tmp6
        tmp8 = tl.math.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhzs3ul6ay5apbvnbglnv5yatb7qjyxri7u22gjmdrmff4leq3j.py
# Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
# x_246 => add_210, add_211, mul_240, mul_241, rsqrt_60, sub_60, var_mean_60
# x_247 => mean
triton_per_fused_mean_native_layer_norm_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (512*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg1_1, (256, ), (1, ))
    assert_size_stride(arg2_1, (256, ), (1, ))
    assert_size_stride(arg3_1, (256, ), (1, ))
    assert_size_stride(arg4_1, (1536, 256), (256, 1))
    assert_size_stride(arg5_1, (1536, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (196, 196), (196, 1))
    assert_size_stride(arg9_1, (196, ), (1, ))
    assert_size_stride(arg10_1, (256, 768), (768, 1))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (1536, 256), (256, 1))
    assert_size_stride(arg15_1, (1536, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (196, 196), (196, 1))
    assert_size_stride(arg19_1, (196, ), (1, ))
    assert_size_stride(arg20_1, (256, 768), (768, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (1536, 256), (256, 1))
    assert_size_stride(arg25_1, (1536, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (196, 196), (196, 1))
    assert_size_stride(arg29_1, (196, ), (1, ))
    assert_size_stride(arg30_1, (256, 768), (768, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (1536, 256), (256, 1))
    assert_size_stride(arg35_1, (1536, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (196, 196), (196, 1))
    assert_size_stride(arg39_1, (196, ), (1, ))
    assert_size_stride(arg40_1, (256, 768), (768, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (1536, 256), (256, 1))
    assert_size_stride(arg45_1, (1536, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (196, 196), (196, 1))
    assert_size_stride(arg49_1, (196, ), (1, ))
    assert_size_stride(arg50_1, (256, 768), (768, 1))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (1536, 256), (256, 1))
    assert_size_stride(arg55_1, (1536, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (196, 196), (196, 1))
    assert_size_stride(arg59_1, (196, ), (1, ))
    assert_size_stride(arg60_1, (256, 768), (768, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (1536, 256), (256, 1))
    assert_size_stride(arg65_1, (1536, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (196, 196), (196, 1))
    assert_size_stride(arg69_1, (196, ), (1, ))
    assert_size_stride(arg70_1, (256, 768), (768, 1))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (1536, 256), (256, 1))
    assert_size_stride(arg75_1, (1536, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (196, 196), (196, 1))
    assert_size_stride(arg79_1, (196, ), (1, ))
    assert_size_stride(arg80_1, (256, 768), (768, 1))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (1536, 256), (256, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (196, 196), (196, 1))
    assert_size_stride(arg89_1, (196, ), (1, ))
    assert_size_stride(arg90_1, (256, 768), (768, 1))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (1536, 256), (256, 1))
    assert_size_stride(arg95_1, (1536, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (196, 196), (196, 1))
    assert_size_stride(arg99_1, (196, ), (1, ))
    assert_size_stride(arg100_1, (256, 768), (768, 1))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (1536, 256), (256, 1))
    assert_size_stride(arg105_1, (1536, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (196, 196), (196, 1))
    assert_size_stride(arg109_1, (196, ), (1, ))
    assert_size_stride(arg110_1, (256, 768), (768, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (1536, 256), (256, 1))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (196, 196), (196, 1))
    assert_size_stride(arg119_1, (196, ), (1, ))
    assert_size_stride(arg120_1, (256, 768), (768, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (1536, 256), (256, 1))
    assert_size_stride(arg125_1, (1536, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (196, 196), (196, 1))
    assert_size_stride(arg129_1, (196, ), (1, ))
    assert_size_stride(arg130_1, (256, 768), (768, 1))
    assert_size_stride(arg131_1, (256, ), (1, ))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (1536, 256), (256, 1))
    assert_size_stride(arg135_1, (1536, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (196, 196), (196, 1))
    assert_size_stride(arg139_1, (196, ), (1, ))
    assert_size_stride(arg140_1, (256, 768), (768, 1))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (1536, 256), (256, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (196, 196), (196, 1))
    assert_size_stride(arg149_1, (196, ), (1, ))
    assert_size_stride(arg150_1, (256, 768), (768, 1))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (1536, 256), (256, 1))
    assert_size_stride(arg155_1, (1536, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (196, 196), (196, 1))
    assert_size_stride(arg159_1, (196, ), (1, ))
    assert_size_stride(arg160_1, (256, 768), (768, 1))
    assert_size_stride(arg161_1, (256, ), (1, ))
    assert_size_stride(arg162_1, (256, ), (1, ))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (1536, 256), (256, 1))
    assert_size_stride(arg165_1, (1536, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (196, 196), (196, 1))
    assert_size_stride(arg169_1, (196, ), (1, ))
    assert_size_stride(arg170_1, (256, 768), (768, 1))
    assert_size_stride(arg171_1, (256, ), (1, ))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (1536, 256), (256, 1))
    assert_size_stride(arg175_1, (1536, ), (1, ))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (196, 196), (196, 1))
    assert_size_stride(arg179_1, (196, ), (1, ))
    assert_size_stride(arg180_1, (256, 768), (768, 1))
    assert_size_stride(arg181_1, (256, ), (1, ))
    assert_size_stride(arg182_1, (256, ), (1, ))
    assert_size_stride(arg183_1, (256, ), (1, ))
    assert_size_stride(arg184_1, (1536, 256), (256, 1))
    assert_size_stride(arg185_1, (1536, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (196, 196), (196, 1))
    assert_size_stride(arg189_1, (196, ), (1, ))
    assert_size_stride(arg190_1, (256, 768), (768, 1))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (1536, 256), (256, 1))
    assert_size_stride(arg195_1, (1536, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (196, 196), (196, 1))
    assert_size_stride(arg199_1, (196, ), (1, ))
    assert_size_stride(arg200_1, (256, 768), (768, 1))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (256, ), (1, ))
    assert_size_stride(arg203_1, (256, ), (1, ))
    assert_size_stride(arg204_1, (1536, 256), (256, 1))
    assert_size_stride(arg205_1, (1536, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (196, 196), (196, 1))
    assert_size_stride(arg209_1, (196, ), (1, ))
    assert_size_stride(arg210_1, (256, 768), (768, 1))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (1536, 256), (256, 1))
    assert_size_stride(arg215_1, (1536, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (196, 196), (196, 1))
    assert_size_stride(arg219_1, (196, ), (1, ))
    assert_size_stride(arg220_1, (256, 768), (768, 1))
    assert_size_stride(arg221_1, (256, ), (1, ))
    assert_size_stride(arg222_1, (256, ), (1, ))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (1536, 256), (256, 1))
    assert_size_stride(arg225_1, (1536, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (196, 196), (196, 1))
    assert_size_stride(arg229_1, (196, ), (1, ))
    assert_size_stride(arg230_1, (256, 768), (768, 1))
    assert_size_stride(arg231_1, (256, ), (1, ))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (1536, 256), (256, 1))
    assert_size_stride(arg235_1, (1536, ), (1, ))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (196, 196), (196, 1))
    assert_size_stride(arg239_1, (196, ), (1, ))
    assert_size_stride(arg240_1, (256, 768), (768, 1))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (1536, 256), (256, 1))
    assert_size_stride(arg245_1, (1536, ), (1, ))
    assert_size_stride(arg246_1, (768, ), (1, ))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (196, 196), (196, 1))
    assert_size_stride(arg249_1, (196, ), (1, ))
    assert_size_stride(arg250_1, (256, 768), (768, 1))
    assert_size_stride(arg251_1, (256, ), (1, ))
    assert_size_stride(arg252_1, (256, ), (1, ))
    assert_size_stride(arg253_1, (256, ), (1, ))
    assert_size_stride(arg254_1, (1536, 256), (256, 1))
    assert_size_stride(arg255_1, (1536, ), (1, ))
    assert_size_stride(arg256_1, (768, ), (1, ))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (196, 196), (196, 1))
    assert_size_stride(arg259_1, (196, ), (1, ))
    assert_size_stride(arg260_1, (256, 768), (768, 1))
    assert_size_stride(arg261_1, (256, ), (1, ))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (1536, 256), (256, 1))
    assert_size_stride(arg265_1, (1536, ), (1, ))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (196, 196), (196, 1))
    assert_size_stride(arg269_1, (196, ), (1, ))
    assert_size_stride(arg270_1, (256, 768), (768, 1))
    assert_size_stride(arg271_1, (256, ), (1, ))
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (256, ), (1, ))
    assert_size_stride(arg274_1, (1536, 256), (256, 1))
    assert_size_stride(arg275_1, (1536, ), (1, ))
    assert_size_stride(arg276_1, (768, ), (1, ))
    assert_size_stride(arg277_1, (768, ), (1, ))
    assert_size_stride(arg278_1, (196, 196), (196, 1))
    assert_size_stride(arg279_1, (196, ), (1, ))
    assert_size_stride(arg280_1, (256, 768), (768, 1))
    assert_size_stride(arg281_1, (256, ), (1, ))
    assert_size_stride(arg282_1, (256, ), (1, ))
    assert_size_stride(arg283_1, (256, ), (1, ))
    assert_size_stride(arg284_1, (1536, 256), (256, 1))
    assert_size_stride(arg285_1, (1536, ), (1, ))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (196, 196), (196, 1))
    assert_size_stride(arg289_1, (196, ), (1, ))
    assert_size_stride(arg290_1, (256, 768), (768, 1))
    assert_size_stride(arg291_1, (256, ), (1, ))
    assert_size_stride(arg292_1, (256, ), (1, ))
    assert_size_stride(arg293_1, (256, ), (1, ))
    assert_size_stride(arg294_1, (1536, 256), (256, 1))
    assert_size_stride(arg295_1, (1536, ), (1, ))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (768, ), (1, ))
    assert_size_stride(arg298_1, (196, 196), (196, 1))
    assert_size_stride(arg299_1, (196, ), (1, ))
    assert_size_stride(arg300_1, (256, 768), (768, 1))
    assert_size_stride(arg301_1, (256, ), (1, ))
    assert_size_stride(arg302_1, (256, ), (1, ))
    assert_size_stride(arg303_1, (256, ), (1, ))
    assert_size_stride(arg304_1, (1000, 256), (256, 1))
    assert_size_stride(arg305_1, (1000, ), (1, ))
    assert_size_stride(arg306_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg306_1, arg0_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg0_1
        del arg306_1
        buf1 = empty_strided((8, 196, 1, 2), (392, 1, 3136, 196), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 2), (392, 1, 3136, 196), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 2), (392, 1, 3136, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg1_1, buf1, buf2, buf3, 3136, 128, grid=grid(3136), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 2, grid=grid(1568), stream=stream0)
        buf7 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_2.run(buf0, arg1_1, buf4, buf5, arg2_1, arg3_1, buf7, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del arg2_1
        del arg3_1
        buf8 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 256), (256, 1), 0), reinterpret_tensor(arg4_1, (256, 1536), (1, 256), 0), out=buf8)
        del arg4_1
        buf9 = buf5; del buf5  # reuse
        buf10 = buf4; del buf4  # reuse
        # Source Nodes: [v_1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf8, arg5_1, buf9, buf10, 1568, 768, grid=grid(1568), stream=stream0)
        buf12 = empty((8, 768, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf8, arg5_1, buf9, buf10, arg6_1, arg7_1, buf12, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg6_1
        del arg7_1
        buf13 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (6144, 196), (196, 1), 0), reinterpret_tensor(arg8_1, (196, 196), (1, 196), 0), out=buf13)
        del arg8_1
        buf14 = reinterpret_tensor(buf12, (8, 196, 768), (150528, 768, 1), 0); del buf12  # reuse
        # Source Nodes: [x_7], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf8, arg5_1, buf13, arg9_1, buf14, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg5_1
        del arg9_1
        buf15 = reinterpret_tensor(buf7, (1568, 256), (256, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf14, (1568, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 256), (1, 768), 0), out=buf15)
        del arg10_1
        buf16 = reinterpret_tensor(buf3, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf3  # reuse
        buf17 = reinterpret_tensor(buf2, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf2  # reuse
        buf18 = reinterpret_tensor(buf1, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf1  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf0, arg1_1, buf15, arg11_1, buf16, buf17, buf18, 3136, 128, grid=grid(3136), stream=stream0)
        buf19 = buf9; del buf9  # reuse
        buf20 = buf10; del buf10  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf16, buf17, buf18, buf19, buf20, 1568, 2, grid=grid(1568), stream=stream0)
        del buf16
        del buf17
        del buf18
        buf22 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_8.run(buf0, arg1_1, buf15, arg11_1, buf19, buf20, arg12_1, arg13_1, buf22, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del arg12_1
        del arg13_1
        buf23 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (1568, 256), (256, 1), 0), reinterpret_tensor(arg14_1, (256, 1536), (1, 256), 0), out=buf23)
        del arg14_1
        buf24 = buf20; del buf20  # reuse
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [v_4], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf23, arg15_1, buf24, buf25, 1568, 768, grid=grid(1568), stream=stream0)
        buf27 = reinterpret_tensor(buf14, (8, 768, 196), (150528, 196, 1), 0); del buf14  # reuse
        # Source Nodes: [v_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf23, arg15_1, buf24, buf25, arg16_1, arg17_1, buf27, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg16_1
        del arg17_1
        buf28 = buf13; del buf13  # reuse
        # Source Nodes: [v_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (6144, 196), (196, 1), 0), reinterpret_tensor(arg18_1, (196, 196), (1, 196), 0), out=buf28)
        del arg18_1
        buf29 = reinterpret_tensor(buf27, (8, 196, 768), (150528, 768, 1), 0); del buf27  # reuse
        # Source Nodes: [x_15], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf23, arg15_1, buf28, arg19_1, buf29, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg15_1
        del arg19_1
        buf30 = reinterpret_tensor(buf22, (1568, 256), (256, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (1568, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 256), (1, 768), 0), out=buf30)
        del arg20_1
        buf31 = reinterpret_tensor(buf15, (8, 196, 256), (50176, 256, 1), 0); del buf15  # reuse
        buf35 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm, x_11, x_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf31, buf0, arg1_1, arg11_1, buf30, arg21_1, arg22_1, arg23_1, buf35, 1568, 256, grid=grid(1568), stream=stream0)
        del arg11_1
        del arg1_1
        del arg21_1
        del arg22_1
        del arg23_1
        buf36 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (1568, 256), (256, 1), 0), reinterpret_tensor(arg24_1, (256, 1536), (1, 256), 0), out=buf36)
        del arg24_1
        buf37 = buf25; del buf25  # reuse
        buf38 = buf24; del buf24  # reuse
        # Source Nodes: [v_7], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf36, arg25_1, buf37, buf38, 1568, 768, grid=grid(1568), stream=stream0)
        buf40 = reinterpret_tensor(buf29, (8, 768, 196), (150528, 196, 1), 0); del buf29  # reuse
        # Source Nodes: [v_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf36, arg25_1, buf37, buf38, arg26_1, arg27_1, buf40, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg26_1
        del arg27_1
        buf41 = buf28; del buf28  # reuse
        # Source Nodes: [v_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (6144, 196), (196, 1), 0), reinterpret_tensor(arg28_1, (196, 196), (1, 196), 0), out=buf41)
        del arg28_1
        buf42 = reinterpret_tensor(buf40, (8, 196, 768), (150528, 768, 1), 0); del buf40  # reuse
        # Source Nodes: [x_23], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf36, arg25_1, buf41, arg29_1, buf42, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg25_1
        del arg29_1
        buf43 = reinterpret_tensor(buf35, (1568, 256), (256, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (1568, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 256), (1, 768), 0), out=buf43)
        del arg30_1
        buf47 = reinterpret_tensor(buf30, (8, 196, 256), (50176, 256, 1), 0); del buf30  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm, x_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf31, buf43, arg31_1, arg32_1, arg33_1, buf47, 1568, 256, grid=grid(1568), stream=stream0)
        del arg32_1
        del arg33_1
        buf48 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (1568, 256), (256, 1), 0), reinterpret_tensor(arg34_1, (256, 1536), (1, 256), 0), out=buf48)
        del arg34_1
        buf49 = buf38; del buf38  # reuse
        buf50 = buf37; del buf37  # reuse
        # Source Nodes: [v_10], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf48, arg35_1, buf49, buf50, 1568, 768, grid=grid(1568), stream=stream0)
        buf52 = reinterpret_tensor(buf42, (8, 768, 196), (150528, 196, 1), 0); del buf42  # reuse
        # Source Nodes: [v_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf48, arg35_1, buf49, buf50, arg36_1, arg37_1, buf52, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg36_1
        del arg37_1
        buf53 = buf41; del buf41  # reuse
        # Source Nodes: [v_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (6144, 196), (196, 1), 0), reinterpret_tensor(arg38_1, (196, 196), (1, 196), 0), out=buf53)
        del arg38_1
        buf54 = reinterpret_tensor(buf52, (8, 196, 768), (150528, 768, 1), 0); del buf52  # reuse
        # Source Nodes: [x_31], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf48, arg35_1, buf53, arg39_1, buf54, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg35_1
        del arg39_1
        buf55 = reinterpret_tensor(buf47, (1568, 256), (256, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (1568, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 256), (1, 768), 0), out=buf55)
        del arg40_1
        buf56 = reinterpret_tensor(buf55, (8, 196, 256), (50176, 256, 1), 0); del buf55  # reuse
        buf60 = reinterpret_tensor(buf0, (8, 196, 256), (50176, 256, 1), 0); del buf0  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm, x_27, x_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf56, buf31, buf43, arg31_1, arg41_1, arg42_1, arg43_1, buf60, 1568, 256, grid=grid(1568), stream=stream0)
        del arg31_1
        del arg41_1
        del arg42_1
        del arg43_1
        buf61 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (1568, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 1536), (1, 256), 0), out=buf61)
        del arg44_1
        buf62 = buf50; del buf50  # reuse
        buf63 = buf49; del buf49  # reuse
        # Source Nodes: [v_13], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf61, arg45_1, buf62, buf63, 1568, 768, grid=grid(1568), stream=stream0)
        buf65 = reinterpret_tensor(buf54, (8, 768, 196), (150528, 196, 1), 0); del buf54  # reuse
        # Source Nodes: [v_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf61, arg45_1, buf62, buf63, arg46_1, arg47_1, buf65, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg46_1
        del arg47_1
        buf66 = buf53; del buf53  # reuse
        # Source Nodes: [v_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (6144, 196), (196, 1), 0), reinterpret_tensor(arg48_1, (196, 196), (1, 196), 0), out=buf66)
        del arg48_1
        buf67 = reinterpret_tensor(buf65, (8, 196, 768), (150528, 768, 1), 0); del buf65  # reuse
        # Source Nodes: [x_39], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf61, arg45_1, buf66, arg49_1, buf67, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg45_1
        del arg49_1
        buf68 = reinterpret_tensor(buf60, (1568, 256), (256, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (1568, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 256), (1, 768), 0), out=buf68)
        del arg50_1
        buf72 = reinterpret_tensor(buf43, (8, 196, 256), (50176, 256, 1), 0); del buf43  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm, x_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf56, buf68, arg51_1, arg52_1, arg53_1, buf72, 1568, 256, grid=grid(1568), stream=stream0)
        del arg52_1
        del arg53_1
        buf73 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (1568, 256), (256, 1), 0), reinterpret_tensor(arg54_1, (256, 1536), (1, 256), 0), out=buf73)
        del arg54_1
        buf74 = buf63; del buf63  # reuse
        buf75 = buf62; del buf62  # reuse
        # Source Nodes: [v_16], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf73, arg55_1, buf74, buf75, 1568, 768, grid=grid(1568), stream=stream0)
        buf77 = reinterpret_tensor(buf67, (8, 768, 196), (150528, 196, 1), 0); del buf67  # reuse
        # Source Nodes: [v_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf73, arg55_1, buf74, buf75, arg56_1, arg57_1, buf77, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg56_1
        del arg57_1
        buf78 = buf66; del buf66  # reuse
        # Source Nodes: [v_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (6144, 196), (196, 1), 0), reinterpret_tensor(arg58_1, (196, 196), (1, 196), 0), out=buf78)
        del arg58_1
        buf79 = reinterpret_tensor(buf77, (8, 196, 768), (150528, 768, 1), 0); del buf77  # reuse
        # Source Nodes: [x_47], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf73, arg55_1, buf78, arg59_1, buf79, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg55_1
        del arg59_1
        buf80 = reinterpret_tensor(buf72, (1568, 256), (256, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (1568, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 256), (1, 768), 0), out=buf80)
        del arg60_1
        buf81 = reinterpret_tensor(buf80, (8, 196, 256), (50176, 256, 1), 0); del buf80  # reuse
        buf85 = buf31; del buf31  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm, x_43, x_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf81, buf56, buf68, arg51_1, arg61_1, arg62_1, arg63_1, buf85, 1568, 256, grid=grid(1568), stream=stream0)
        del arg51_1
        del arg61_1
        del arg62_1
        del arg63_1
        buf86 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (1568, 256), (256, 1), 0), reinterpret_tensor(arg64_1, (256, 1536), (1, 256), 0), out=buf86)
        del arg64_1
        buf87 = buf75; del buf75  # reuse
        buf88 = buf74; del buf74  # reuse
        # Source Nodes: [v_19], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf86, arg65_1, buf87, buf88, 1568, 768, grid=grid(1568), stream=stream0)
        buf90 = reinterpret_tensor(buf79, (8, 768, 196), (150528, 196, 1), 0); del buf79  # reuse
        # Source Nodes: [v_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf86, arg65_1, buf87, buf88, arg66_1, arg67_1, buf90, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg66_1
        del arg67_1
        buf91 = buf78; del buf78  # reuse
        # Source Nodes: [v_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (6144, 196), (196, 1), 0), reinterpret_tensor(arg68_1, (196, 196), (1, 196), 0), out=buf91)
        del arg68_1
        buf92 = reinterpret_tensor(buf90, (8, 196, 768), (150528, 768, 1), 0); del buf90  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf86, arg65_1, buf91, arg69_1, buf92, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg65_1
        del arg69_1
        buf93 = reinterpret_tensor(buf85, (1568, 256), (256, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1568, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 256), (1, 768), 0), out=buf93)
        del arg70_1
        buf97 = reinterpret_tensor(buf68, (8, 196, 256), (50176, 256, 1), 0); del buf68  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm, x_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf81, buf93, arg71_1, arg72_1, arg73_1, buf97, 1568, 256, grid=grid(1568), stream=stream0)
        del arg72_1
        del arg73_1
        buf98 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (1568, 256), (256, 1), 0), reinterpret_tensor(arg74_1, (256, 1536), (1, 256), 0), out=buf98)
        del arg74_1
        buf99 = buf88; del buf88  # reuse
        buf100 = buf87; del buf87  # reuse
        # Source Nodes: [v_22], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf98, arg75_1, buf99, buf100, 1568, 768, grid=grid(1568), stream=stream0)
        buf102 = reinterpret_tensor(buf92, (8, 768, 196), (150528, 196, 1), 0); del buf92  # reuse
        # Source Nodes: [v_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf98, arg75_1, buf99, buf100, arg76_1, arg77_1, buf102, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg76_1
        del arg77_1
        buf103 = buf91; del buf91  # reuse
        # Source Nodes: [v_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (6144, 196), (196, 1), 0), reinterpret_tensor(arg78_1, (196, 196), (1, 196), 0), out=buf103)
        del arg78_1
        buf104 = reinterpret_tensor(buf102, (8, 196, 768), (150528, 768, 1), 0); del buf102  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf98, arg75_1, buf103, arg79_1, buf104, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg75_1
        del arg79_1
        buf105 = reinterpret_tensor(buf97, (1568, 256), (256, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (1568, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 256), (1, 768), 0), out=buf105)
        del arg80_1
        buf106 = reinterpret_tensor(buf105, (8, 196, 256), (50176, 256, 1), 0); del buf105  # reuse
        buf110 = buf56; del buf56  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm, x_59, x_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf106, buf81, buf93, arg71_1, arg81_1, arg82_1, arg83_1, buf110, 1568, 256, grid=grid(1568), stream=stream0)
        del arg71_1
        del arg81_1
        del arg82_1
        del arg83_1
        buf111 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (1568, 256), (256, 1), 0), reinterpret_tensor(arg84_1, (256, 1536), (1, 256), 0), out=buf111)
        del arg84_1
        buf112 = buf99; del buf99  # reuse
        buf113 = buf100; del buf100  # reuse
        # Source Nodes: [v_25], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf111, arg85_1, buf112, buf113, 1568, 768, grid=grid(1568), stream=stream0)
        buf115 = reinterpret_tensor(buf104, (8, 768, 196), (150528, 196, 1), 0); del buf104  # reuse
        # Source Nodes: [v_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf111, arg85_1, buf112, buf113, arg86_1, arg87_1, buf115, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg86_1
        del arg87_1
        buf116 = buf103; del buf103  # reuse
        # Source Nodes: [v_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (6144, 196), (196, 1), 0), reinterpret_tensor(arg88_1, (196, 196), (1, 196), 0), out=buf116)
        del arg88_1
        buf117 = reinterpret_tensor(buf115, (8, 196, 768), (150528, 768, 1), 0); del buf115  # reuse
        # Source Nodes: [x_71], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf111, arg85_1, buf116, arg89_1, buf117, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg85_1
        del arg89_1
        buf118 = reinterpret_tensor(buf110, (1568, 256), (256, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (1568, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 256), (1, 768), 0), out=buf118)
        del arg90_1
        buf122 = reinterpret_tensor(buf93, (8, 196, 256), (50176, 256, 1), 0); del buf93  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm, x_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf106, buf118, arg91_1, arg92_1, arg93_1, buf122, 1568, 256, grid=grid(1568), stream=stream0)
        del arg92_1
        del arg93_1
        buf123 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 1536), (1, 256), 0), out=buf123)
        del arg94_1
        buf124 = buf113; del buf113  # reuse
        buf125 = buf112; del buf112  # reuse
        # Source Nodes: [v_28], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf123, arg95_1, buf124, buf125, 1568, 768, grid=grid(1568), stream=stream0)
        buf127 = reinterpret_tensor(buf117, (8, 768, 196), (150528, 196, 1), 0); del buf117  # reuse
        # Source Nodes: [v_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf123, arg95_1, buf124, buf125, arg96_1, arg97_1, buf127, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg96_1
        del arg97_1
        buf128 = buf116; del buf116  # reuse
        # Source Nodes: [v_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (6144, 196), (196, 1), 0), reinterpret_tensor(arg98_1, (196, 196), (1, 196), 0), out=buf128)
        del arg98_1
        buf129 = reinterpret_tensor(buf127, (8, 196, 768), (150528, 768, 1), 0); del buf127  # reuse
        # Source Nodes: [x_79], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf123, arg95_1, buf128, arg99_1, buf129, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg95_1
        del arg99_1
        buf130 = reinterpret_tensor(buf122, (1568, 256), (256, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (1568, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 256), (1, 768), 0), out=buf130)
        del arg100_1
        buf131 = reinterpret_tensor(buf130, (8, 196, 256), (50176, 256, 1), 0); del buf130  # reuse
        buf135 = buf81; del buf81  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm, x_75, x_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf131, buf106, buf118, arg91_1, arg101_1, arg102_1, arg103_1, buf135, 1568, 256, grid=grid(1568), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg91_1
        buf136 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 256), (256, 1), 0), reinterpret_tensor(arg104_1, (256, 1536), (1, 256), 0), out=buf136)
        del arg104_1
        buf137 = buf125; del buf125  # reuse
        buf138 = buf124; del buf124  # reuse
        # Source Nodes: [v_31], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf136, arg105_1, buf137, buf138, 1568, 768, grid=grid(1568), stream=stream0)
        buf140 = reinterpret_tensor(buf129, (8, 768, 196), (150528, 196, 1), 0); del buf129  # reuse
        # Source Nodes: [v_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf136, arg105_1, buf137, buf138, arg106_1, arg107_1, buf140, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg106_1
        del arg107_1
        buf141 = buf128; del buf128  # reuse
        # Source Nodes: [v_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (6144, 196), (196, 1), 0), reinterpret_tensor(arg108_1, (196, 196), (1, 196), 0), out=buf141)
        del arg108_1
        buf142 = reinterpret_tensor(buf140, (8, 196, 768), (150528, 768, 1), 0); del buf140  # reuse
        # Source Nodes: [x_87], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf136, arg105_1, buf141, arg109_1, buf142, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg105_1
        del arg109_1
        buf143 = reinterpret_tensor(buf135, (1568, 256), (256, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1568, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 256), (1, 768), 0), out=buf143)
        del arg110_1
        buf147 = reinterpret_tensor(buf118, (8, 196, 256), (50176, 256, 1), 0); del buf118  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm, x_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf131, buf143, arg111_1, arg112_1, arg113_1, buf147, 1568, 256, grid=grid(1568), stream=stream0)
        del arg112_1
        del arg113_1
        buf148 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1568, 256), (256, 1), 0), reinterpret_tensor(arg114_1, (256, 1536), (1, 256), 0), out=buf148)
        del arg114_1
        buf149 = buf138; del buf138  # reuse
        buf150 = buf137; del buf137  # reuse
        # Source Nodes: [v_34], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf148, arg115_1, buf149, buf150, 1568, 768, grid=grid(1568), stream=stream0)
        buf152 = reinterpret_tensor(buf142, (8, 768, 196), (150528, 196, 1), 0); del buf142  # reuse
        # Source Nodes: [v_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf148, arg115_1, buf149, buf150, arg116_1, arg117_1, buf152, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg116_1
        del arg117_1
        buf153 = buf141; del buf141  # reuse
        # Source Nodes: [v_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (6144, 196), (196, 1), 0), reinterpret_tensor(arg118_1, (196, 196), (1, 196), 0), out=buf153)
        del arg118_1
        buf154 = reinterpret_tensor(buf152, (8, 196, 768), (150528, 768, 1), 0); del buf152  # reuse
        # Source Nodes: [x_95], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf148, arg115_1, buf153, arg119_1, buf154, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg115_1
        del arg119_1
        buf155 = reinterpret_tensor(buf147, (1568, 256), (256, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (1568, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 256), (1, 768), 0), out=buf155)
        del arg120_1
        buf156 = reinterpret_tensor(buf155, (8, 196, 256), (50176, 256, 1), 0); del buf155  # reuse
        buf160 = buf106; del buf106  # reuse
        # Source Nodes: [getattr_l__mod___blocks___12___norm, x_91, x_99], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf156, buf131, buf143, arg111_1, arg121_1, arg122_1, arg123_1, buf160, 1568, 256, grid=grid(1568), stream=stream0)
        del arg111_1
        del arg121_1
        del arg122_1
        del arg123_1
        buf161 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (1568, 256), (256, 1), 0), reinterpret_tensor(arg124_1, (256, 1536), (1, 256), 0), out=buf161)
        del arg124_1
        buf162 = buf150; del buf150  # reuse
        buf163 = buf149; del buf149  # reuse
        # Source Nodes: [v_37], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf161, arg125_1, buf162, buf163, 1568, 768, grid=grid(1568), stream=stream0)
        buf165 = reinterpret_tensor(buf154, (8, 768, 196), (150528, 196, 1), 0); del buf154  # reuse
        # Source Nodes: [v_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf161, arg125_1, buf162, buf163, arg126_1, arg127_1, buf165, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg126_1
        del arg127_1
        buf166 = buf153; del buf153  # reuse
        # Source Nodes: [v_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (6144, 196), (196, 1), 0), reinterpret_tensor(arg128_1, (196, 196), (1, 196), 0), out=buf166)
        del arg128_1
        buf167 = reinterpret_tensor(buf165, (8, 196, 768), (150528, 768, 1), 0); del buf165  # reuse
        # Source Nodes: [x_103], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf161, arg125_1, buf166, arg129_1, buf167, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg125_1
        del arg129_1
        buf168 = reinterpret_tensor(buf160, (1568, 256), (256, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (1568, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 256), (1, 768), 0), out=buf168)
        del arg130_1
        buf172 = reinterpret_tensor(buf143, (8, 196, 256), (50176, 256, 1), 0); del buf143  # reuse
        # Source Nodes: [getattr_l__mod___blocks___13___norm, x_107], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf156, buf168, arg131_1, arg132_1, arg133_1, buf172, 1568, 256, grid=grid(1568), stream=stream0)
        del arg132_1
        del arg133_1
        buf173 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (1568, 256), (256, 1), 0), reinterpret_tensor(arg134_1, (256, 1536), (1, 256), 0), out=buf173)
        del arg134_1
        buf174 = buf163; del buf163  # reuse
        buf175 = buf162; del buf162  # reuse
        # Source Nodes: [v_40], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf173, arg135_1, buf174, buf175, 1568, 768, grid=grid(1568), stream=stream0)
        buf177 = reinterpret_tensor(buf167, (8, 768, 196), (150528, 196, 1), 0); del buf167  # reuse
        # Source Nodes: [v_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf173, arg135_1, buf174, buf175, arg136_1, arg137_1, buf177, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg136_1
        del arg137_1
        buf178 = buf166; del buf166  # reuse
        # Source Nodes: [v_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (6144, 196), (196, 1), 0), reinterpret_tensor(arg138_1, (196, 196), (1, 196), 0), out=buf178)
        del arg138_1
        buf179 = reinterpret_tensor(buf177, (8, 196, 768), (150528, 768, 1), 0); del buf177  # reuse
        # Source Nodes: [x_111], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf173, arg135_1, buf178, arg139_1, buf179, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg135_1
        del arg139_1
        buf180 = reinterpret_tensor(buf172, (1568, 256), (256, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (1568, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 256), (1, 768), 0), out=buf180)
        del arg140_1
        buf181 = reinterpret_tensor(buf180, (8, 196, 256), (50176, 256, 1), 0); del buf180  # reuse
        buf185 = buf131; del buf131  # reuse
        # Source Nodes: [getattr_l__mod___blocks___14___norm, x_107, x_115], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf181, buf156, buf168, arg131_1, arg141_1, arg142_1, arg143_1, buf185, 1568, 256, grid=grid(1568), stream=stream0)
        del arg131_1
        del arg141_1
        del arg142_1
        del arg143_1
        buf186 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (1568, 256), (256, 1), 0), reinterpret_tensor(arg144_1, (256, 1536), (1, 256), 0), out=buf186)
        del arg144_1
        buf187 = buf175; del buf175  # reuse
        buf188 = buf174; del buf174  # reuse
        # Source Nodes: [v_43], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf186, arg145_1, buf187, buf188, 1568, 768, grid=grid(1568), stream=stream0)
        buf190 = reinterpret_tensor(buf179, (8, 768, 196), (150528, 196, 1), 0); del buf179  # reuse
        # Source Nodes: [v_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf186, arg145_1, buf187, buf188, arg146_1, arg147_1, buf190, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg146_1
        del arg147_1
        buf191 = buf178; del buf178  # reuse
        # Source Nodes: [v_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (6144, 196), (196, 1), 0), reinterpret_tensor(arg148_1, (196, 196), (1, 196), 0), out=buf191)
        del arg148_1
        buf192 = reinterpret_tensor(buf190, (8, 196, 768), (150528, 768, 1), 0); del buf190  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf186, arg145_1, buf191, arg149_1, buf192, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg145_1
        del arg149_1
        buf193 = reinterpret_tensor(buf185, (1568, 256), (256, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (1568, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 256), (1, 768), 0), out=buf193)
        del arg150_1
        buf197 = reinterpret_tensor(buf168, (8, 196, 256), (50176, 256, 1), 0); del buf168  # reuse
        # Source Nodes: [getattr_l__mod___blocks___15___norm, x_123], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf181, buf193, arg151_1, arg152_1, arg153_1, buf197, 1568, 256, grid=grid(1568), stream=stream0)
        del arg152_1
        del arg153_1
        buf198 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (1568, 256), (256, 1), 0), reinterpret_tensor(arg154_1, (256, 1536), (1, 256), 0), out=buf198)
        del arg154_1
        buf199 = buf188; del buf188  # reuse
        buf200 = buf187; del buf187  # reuse
        # Source Nodes: [v_46], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf198, arg155_1, buf199, buf200, 1568, 768, grid=grid(1568), stream=stream0)
        buf202 = reinterpret_tensor(buf192, (8, 768, 196), (150528, 196, 1), 0); del buf192  # reuse
        # Source Nodes: [v_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf198, arg155_1, buf199, buf200, arg156_1, arg157_1, buf202, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg156_1
        del arg157_1
        buf203 = buf191; del buf191  # reuse
        # Source Nodes: [v_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (6144, 196), (196, 1), 0), reinterpret_tensor(arg158_1, (196, 196), (1, 196), 0), out=buf203)
        del arg158_1
        buf204 = reinterpret_tensor(buf202, (8, 196, 768), (150528, 768, 1), 0); del buf202  # reuse
        # Source Nodes: [x_127], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf198, arg155_1, buf203, arg159_1, buf204, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg155_1
        del arg159_1
        buf205 = reinterpret_tensor(buf197, (1568, 256), (256, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (1568, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 256), (1, 768), 0), out=buf205)
        del arg160_1
        buf206 = reinterpret_tensor(buf205, (8, 196, 256), (50176, 256, 1), 0); del buf205  # reuse
        buf210 = buf156; del buf156  # reuse
        # Source Nodes: [getattr_l__mod___blocks___16___norm, x_123, x_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf206, buf181, buf193, arg151_1, arg161_1, arg162_1, arg163_1, buf210, 1568, 256, grid=grid(1568), stream=stream0)
        del arg151_1
        del arg161_1
        del arg162_1
        del arg163_1
        buf211 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (1568, 256), (256, 1), 0), reinterpret_tensor(arg164_1, (256, 1536), (1, 256), 0), out=buf211)
        del arg164_1
        buf212 = buf200; del buf200  # reuse
        buf213 = buf199; del buf199  # reuse
        # Source Nodes: [v_49], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf211, arg165_1, buf212, buf213, 1568, 768, grid=grid(1568), stream=stream0)
        buf215 = reinterpret_tensor(buf204, (8, 768, 196), (150528, 196, 1), 0); del buf204  # reuse
        # Source Nodes: [v_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf211, arg165_1, buf212, buf213, arg166_1, arg167_1, buf215, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg166_1
        del arg167_1
        buf216 = buf203; del buf203  # reuse
        # Source Nodes: [v_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (6144, 196), (196, 1), 0), reinterpret_tensor(arg168_1, (196, 196), (1, 196), 0), out=buf216)
        del arg168_1
        buf217 = reinterpret_tensor(buf215, (8, 196, 768), (150528, 768, 1), 0); del buf215  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf211, arg165_1, buf216, arg169_1, buf217, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg165_1
        del arg169_1
        buf218 = reinterpret_tensor(buf210, (1568, 256), (256, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (1568, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 256), (1, 768), 0), out=buf218)
        del arg170_1
        buf222 = reinterpret_tensor(buf193, (8, 196, 256), (50176, 256, 1), 0); del buf193  # reuse
        # Source Nodes: [getattr_l__mod___blocks___17___norm, x_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf206, buf218, arg171_1, arg172_1, arg173_1, buf222, 1568, 256, grid=grid(1568), stream=stream0)
        del arg172_1
        del arg173_1
        buf223 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (1568, 256), (256, 1), 0), reinterpret_tensor(arg174_1, (256, 1536), (1, 256), 0), out=buf223)
        del arg174_1
        buf224 = buf213; del buf213  # reuse
        buf225 = buf212; del buf212  # reuse
        # Source Nodes: [v_52], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf223, arg175_1, buf224, buf225, 1568, 768, grid=grid(1568), stream=stream0)
        buf227 = reinterpret_tensor(buf217, (8, 768, 196), (150528, 196, 1), 0); del buf217  # reuse
        # Source Nodes: [v_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf223, arg175_1, buf224, buf225, arg176_1, arg177_1, buf227, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg176_1
        del arg177_1
        buf228 = buf216; del buf216  # reuse
        # Source Nodes: [v_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (6144, 196), (196, 1), 0), reinterpret_tensor(arg178_1, (196, 196), (1, 196), 0), out=buf228)
        del arg178_1
        buf229 = reinterpret_tensor(buf227, (8, 196, 768), (150528, 768, 1), 0); del buf227  # reuse
        # Source Nodes: [x_143], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf223, arg175_1, buf228, arg179_1, buf229, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg175_1
        del arg179_1
        buf230 = reinterpret_tensor(buf222, (1568, 256), (256, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1568, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 256), (1, 768), 0), out=buf230)
        del arg180_1
        buf231 = reinterpret_tensor(buf230, (8, 196, 256), (50176, 256, 1), 0); del buf230  # reuse
        buf235 = buf181; del buf181  # reuse
        # Source Nodes: [getattr_l__mod___blocks___18___norm, x_139, x_147], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf231, buf206, buf218, arg171_1, arg181_1, arg182_1, arg183_1, buf235, 1568, 256, grid=grid(1568), stream=stream0)
        del arg171_1
        del arg181_1
        del arg182_1
        del arg183_1
        buf236 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (1568, 256), (256, 1), 0), reinterpret_tensor(arg184_1, (256, 1536), (1, 256), 0), out=buf236)
        del arg184_1
        buf237 = buf225; del buf225  # reuse
        buf238 = buf224; del buf224  # reuse
        # Source Nodes: [v_55], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf236, arg185_1, buf237, buf238, 1568, 768, grid=grid(1568), stream=stream0)
        buf240 = reinterpret_tensor(buf229, (8, 768, 196), (150528, 196, 1), 0); del buf229  # reuse
        # Source Nodes: [v_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf236, arg185_1, buf237, buf238, arg186_1, arg187_1, buf240, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg186_1
        del arg187_1
        buf241 = buf228; del buf228  # reuse
        # Source Nodes: [v_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (6144, 196), (196, 1), 0), reinterpret_tensor(arg188_1, (196, 196), (1, 196), 0), out=buf241)
        del arg188_1
        buf242 = reinterpret_tensor(buf240, (8, 196, 768), (150528, 768, 1), 0); del buf240  # reuse
        # Source Nodes: [x_151], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf236, arg185_1, buf241, arg189_1, buf242, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg185_1
        del arg189_1
        buf243 = reinterpret_tensor(buf235, (1568, 256), (256, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 768), (768, 1), 0), reinterpret_tensor(arg190_1, (768, 256), (1, 768), 0), out=buf243)
        del arg190_1
        buf247 = reinterpret_tensor(buf218, (8, 196, 256), (50176, 256, 1), 0); del buf218  # reuse
        # Source Nodes: [getattr_l__mod___blocks___19___norm, x_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf231, buf243, arg191_1, arg192_1, arg193_1, buf247, 1568, 256, grid=grid(1568), stream=stream0)
        del arg192_1
        del arg193_1
        buf248 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (1568, 256), (256, 1), 0), reinterpret_tensor(arg194_1, (256, 1536), (1, 256), 0), out=buf248)
        del arg194_1
        buf249 = buf238; del buf238  # reuse
        buf250 = buf237; del buf237  # reuse
        # Source Nodes: [v_58], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf248, arg195_1, buf249, buf250, 1568, 768, grid=grid(1568), stream=stream0)
        buf252 = reinterpret_tensor(buf242, (8, 768, 196), (150528, 196, 1), 0); del buf242  # reuse
        # Source Nodes: [v_59], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf248, arg195_1, buf249, buf250, arg196_1, arg197_1, buf252, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg196_1
        del arg197_1
        buf253 = buf241; del buf241  # reuse
        # Source Nodes: [v_59], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (6144, 196), (196, 1), 0), reinterpret_tensor(arg198_1, (196, 196), (1, 196), 0), out=buf253)
        del arg198_1
        buf254 = reinterpret_tensor(buf252, (8, 196, 768), (150528, 768, 1), 0); del buf252  # reuse
        # Source Nodes: [x_159], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf248, arg195_1, buf253, arg199_1, buf254, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg195_1
        del arg199_1
        buf255 = reinterpret_tensor(buf247, (1568, 256), (256, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf254, (1568, 768), (768, 1), 0), reinterpret_tensor(arg200_1, (768, 256), (1, 768), 0), out=buf255)
        del arg200_1
        buf256 = reinterpret_tensor(buf255, (8, 196, 256), (50176, 256, 1), 0); del buf255  # reuse
        buf260 = buf206; del buf206  # reuse
        # Source Nodes: [getattr_l__mod___blocks___20___norm, x_155, x_163], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf256, buf231, buf243, arg191_1, arg201_1, arg202_1, arg203_1, buf260, 1568, 256, grid=grid(1568), stream=stream0)
        del arg191_1
        del arg201_1
        del arg202_1
        del arg203_1
        buf261 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (1568, 256), (256, 1), 0), reinterpret_tensor(arg204_1, (256, 1536), (1, 256), 0), out=buf261)
        del arg204_1
        buf262 = buf250; del buf250  # reuse
        buf263 = buf249; del buf249  # reuse
        # Source Nodes: [v_61], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf261, arg205_1, buf262, buf263, 1568, 768, grid=grid(1568), stream=stream0)
        buf265 = reinterpret_tensor(buf254, (8, 768, 196), (150528, 196, 1), 0); del buf254  # reuse
        # Source Nodes: [v_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf261, arg205_1, buf262, buf263, arg206_1, arg207_1, buf265, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg206_1
        del arg207_1
        buf266 = buf253; del buf253  # reuse
        # Source Nodes: [v_62], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (6144, 196), (196, 1), 0), reinterpret_tensor(arg208_1, (196, 196), (1, 196), 0), out=buf266)
        del arg208_1
        buf267 = reinterpret_tensor(buf265, (8, 196, 768), (150528, 768, 1), 0); del buf265  # reuse
        # Source Nodes: [x_167], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf261, arg205_1, buf266, arg209_1, buf267, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg205_1
        del arg209_1
        buf268 = reinterpret_tensor(buf260, (1568, 256), (256, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1568, 768), (768, 1), 0), reinterpret_tensor(arg210_1, (768, 256), (1, 768), 0), out=buf268)
        del arg210_1
        buf272 = reinterpret_tensor(buf243, (8, 196, 256), (50176, 256, 1), 0); del buf243  # reuse
        # Source Nodes: [getattr_l__mod___blocks___21___norm, x_171], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf256, buf268, arg211_1, arg212_1, arg213_1, buf272, 1568, 256, grid=grid(1568), stream=stream0)
        del arg212_1
        del arg213_1
        buf273 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (1568, 256), (256, 1), 0), reinterpret_tensor(arg214_1, (256, 1536), (1, 256), 0), out=buf273)
        del arg214_1
        buf274 = buf263; del buf263  # reuse
        buf275 = buf262; del buf262  # reuse
        # Source Nodes: [v_64], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf273, arg215_1, buf274, buf275, 1568, 768, grid=grid(1568), stream=stream0)
        buf277 = reinterpret_tensor(buf267, (8, 768, 196), (150528, 196, 1), 0); del buf267  # reuse
        # Source Nodes: [v_65], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf273, arg215_1, buf274, buf275, arg216_1, arg217_1, buf277, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg216_1
        del arg217_1
        buf278 = buf266; del buf266  # reuse
        # Source Nodes: [v_65], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (6144, 196), (196, 1), 0), reinterpret_tensor(arg218_1, (196, 196), (1, 196), 0), out=buf278)
        del arg218_1
        buf279 = reinterpret_tensor(buf277, (8, 196, 768), (150528, 768, 1), 0); del buf277  # reuse
        # Source Nodes: [x_175], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf273, arg215_1, buf278, arg219_1, buf279, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg215_1
        del arg219_1
        buf280 = reinterpret_tensor(buf272, (1568, 256), (256, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (1568, 768), (768, 1), 0), reinterpret_tensor(arg220_1, (768, 256), (1, 768), 0), out=buf280)
        del arg220_1
        buf281 = reinterpret_tensor(buf280, (8, 196, 256), (50176, 256, 1), 0); del buf280  # reuse
        buf285 = buf231; del buf231  # reuse
        # Source Nodes: [getattr_l__mod___blocks___22___norm, x_171, x_179], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf281, buf256, buf268, arg211_1, arg221_1, arg222_1, arg223_1, buf285, 1568, 256, grid=grid(1568), stream=stream0)
        del arg211_1
        del arg221_1
        del arg222_1
        del arg223_1
        buf286 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf285, (1568, 256), (256, 1), 0), reinterpret_tensor(arg224_1, (256, 1536), (1, 256), 0), out=buf286)
        del arg224_1
        buf287 = buf275; del buf275  # reuse
        buf288 = buf274; del buf274  # reuse
        # Source Nodes: [v_67], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf286, arg225_1, buf287, buf288, 1568, 768, grid=grid(1568), stream=stream0)
        buf290 = reinterpret_tensor(buf279, (8, 768, 196), (150528, 196, 1), 0); del buf279  # reuse
        # Source Nodes: [v_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf286, arg225_1, buf287, buf288, arg226_1, arg227_1, buf290, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg226_1
        del arg227_1
        buf291 = buf278; del buf278  # reuse
        # Source Nodes: [v_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (6144, 196), (196, 1), 0), reinterpret_tensor(arg228_1, (196, 196), (1, 196), 0), out=buf291)
        del arg228_1
        buf292 = reinterpret_tensor(buf290, (8, 196, 768), (150528, 768, 1), 0); del buf290  # reuse
        # Source Nodes: [x_183], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf286, arg225_1, buf291, arg229_1, buf292, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg225_1
        del arg229_1
        buf293 = reinterpret_tensor(buf285, (1568, 256), (256, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (1568, 768), (768, 1), 0), reinterpret_tensor(arg230_1, (768, 256), (1, 768), 0), out=buf293)
        del arg230_1
        buf297 = reinterpret_tensor(buf268, (8, 196, 256), (50176, 256, 1), 0); del buf268  # reuse
        # Source Nodes: [getattr_l__mod___blocks___23___norm, x_187], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf281, buf293, arg231_1, arg232_1, arg233_1, buf297, 1568, 256, grid=grid(1568), stream=stream0)
        del arg232_1
        del arg233_1
        buf298 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (1568, 256), (256, 1), 0), reinterpret_tensor(arg234_1, (256, 1536), (1, 256), 0), out=buf298)
        del arg234_1
        buf299 = buf288; del buf288  # reuse
        buf300 = buf287; del buf287  # reuse
        # Source Nodes: [v_70], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf298, arg235_1, buf299, buf300, 1568, 768, grid=grid(1568), stream=stream0)
        buf302 = reinterpret_tensor(buf292, (8, 768, 196), (150528, 196, 1), 0); del buf292  # reuse
        # Source Nodes: [v_71], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf298, arg235_1, buf299, buf300, arg236_1, arg237_1, buf302, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg236_1
        del arg237_1
        buf303 = buf291; del buf291  # reuse
        # Source Nodes: [v_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (6144, 196), (196, 1), 0), reinterpret_tensor(arg238_1, (196, 196), (1, 196), 0), out=buf303)
        del arg238_1
        buf304 = reinterpret_tensor(buf302, (8, 196, 768), (150528, 768, 1), 0); del buf302  # reuse
        # Source Nodes: [x_191], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf298, arg235_1, buf303, arg239_1, buf304, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg235_1
        del arg239_1
        buf305 = reinterpret_tensor(buf297, (1568, 256), (256, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (1568, 768), (768, 1), 0), reinterpret_tensor(arg240_1, (768, 256), (1, 768), 0), out=buf305)
        del arg240_1
        buf306 = reinterpret_tensor(buf305, (8, 196, 256), (50176, 256, 1), 0); del buf305  # reuse
        buf310 = buf256; del buf256  # reuse
        # Source Nodes: [getattr_l__mod___blocks___24___norm, x_187, x_195], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf306, buf281, buf293, arg231_1, arg241_1, arg242_1, arg243_1, buf310, 1568, 256, grid=grid(1568), stream=stream0)
        del arg231_1
        del arg241_1
        del arg242_1
        del arg243_1
        buf311 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 256), (256, 1), 0), reinterpret_tensor(arg244_1, (256, 1536), (1, 256), 0), out=buf311)
        del arg244_1
        buf312 = buf300; del buf300  # reuse
        buf313 = buf299; del buf299  # reuse
        # Source Nodes: [v_73], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf311, arg245_1, buf312, buf313, 1568, 768, grid=grid(1568), stream=stream0)
        buf315 = reinterpret_tensor(buf304, (8, 768, 196), (150528, 196, 1), 0); del buf304  # reuse
        # Source Nodes: [v_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf311, arg245_1, buf312, buf313, arg246_1, arg247_1, buf315, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg246_1
        del arg247_1
        buf316 = buf303; del buf303  # reuse
        # Source Nodes: [v_74], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (6144, 196), (196, 1), 0), reinterpret_tensor(arg248_1, (196, 196), (1, 196), 0), out=buf316)
        del arg248_1
        buf317 = reinterpret_tensor(buf315, (8, 196, 768), (150528, 768, 1), 0); del buf315  # reuse
        # Source Nodes: [x_199], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf311, arg245_1, buf316, arg249_1, buf317, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg245_1
        del arg249_1
        buf318 = reinterpret_tensor(buf310, (1568, 256), (256, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (1568, 768), (768, 1), 0), reinterpret_tensor(arg250_1, (768, 256), (1, 768), 0), out=buf318)
        del arg250_1
        buf322 = reinterpret_tensor(buf293, (8, 196, 256), (50176, 256, 1), 0); del buf293  # reuse
        # Source Nodes: [getattr_l__mod___blocks___25___norm, x_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf306, buf318, arg251_1, arg252_1, arg253_1, buf322, 1568, 256, grid=grid(1568), stream=stream0)
        del arg252_1
        del arg253_1
        buf323 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (1568, 256), (256, 1), 0), reinterpret_tensor(arg254_1, (256, 1536), (1, 256), 0), out=buf323)
        del arg254_1
        buf324 = buf313; del buf313  # reuse
        buf325 = buf312; del buf312  # reuse
        # Source Nodes: [v_76], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf323, arg255_1, buf324, buf325, 1568, 768, grid=grid(1568), stream=stream0)
        buf327 = reinterpret_tensor(buf317, (8, 768, 196), (150528, 196, 1), 0); del buf317  # reuse
        # Source Nodes: [v_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf323, arg255_1, buf324, buf325, arg256_1, arg257_1, buf327, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg256_1
        del arg257_1
        buf328 = buf316; del buf316  # reuse
        # Source Nodes: [v_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (6144, 196), (196, 1), 0), reinterpret_tensor(arg258_1, (196, 196), (1, 196), 0), out=buf328)
        del arg258_1
        buf329 = reinterpret_tensor(buf327, (8, 196, 768), (150528, 768, 1), 0); del buf327  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf323, arg255_1, buf328, arg259_1, buf329, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg255_1
        del arg259_1
        buf330 = reinterpret_tensor(buf322, (1568, 256), (256, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (1568, 768), (768, 1), 0), reinterpret_tensor(arg260_1, (768, 256), (1, 768), 0), out=buf330)
        del arg260_1
        buf331 = reinterpret_tensor(buf330, (8, 196, 256), (50176, 256, 1), 0); del buf330  # reuse
        buf335 = buf281; del buf281  # reuse
        # Source Nodes: [getattr_l__mod___blocks___26___norm, x_203, x_211], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf331, buf306, buf318, arg251_1, arg261_1, arg262_1, arg263_1, buf335, 1568, 256, grid=grid(1568), stream=stream0)
        del arg251_1
        del arg261_1
        del arg262_1
        del arg263_1
        buf336 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1568, 256), (256, 1), 0), reinterpret_tensor(arg264_1, (256, 1536), (1, 256), 0), out=buf336)
        del arg264_1
        buf337 = buf325; del buf325  # reuse
        buf338 = buf324; del buf324  # reuse
        # Source Nodes: [v_79], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf336, arg265_1, buf337, buf338, 1568, 768, grid=grid(1568), stream=stream0)
        buf340 = reinterpret_tensor(buf329, (8, 768, 196), (150528, 196, 1), 0); del buf329  # reuse
        # Source Nodes: [v_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf336, arg265_1, buf337, buf338, arg266_1, arg267_1, buf340, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg266_1
        del arg267_1
        buf341 = buf328; del buf328  # reuse
        # Source Nodes: [v_80], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (6144, 196), (196, 1), 0), reinterpret_tensor(arg268_1, (196, 196), (1, 196), 0), out=buf341)
        del arg268_1
        buf342 = reinterpret_tensor(buf340, (8, 196, 768), (150528, 768, 1), 0); del buf340  # reuse
        # Source Nodes: [x_215], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf336, arg265_1, buf341, arg269_1, buf342, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg265_1
        del arg269_1
        buf343 = reinterpret_tensor(buf335, (1568, 256), (256, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (1568, 768), (768, 1), 0), reinterpret_tensor(arg270_1, (768, 256), (1, 768), 0), out=buf343)
        del arg270_1
        buf347 = reinterpret_tensor(buf318, (8, 196, 256), (50176, 256, 1), 0); del buf318  # reuse
        # Source Nodes: [getattr_l__mod___blocks___27___norm, x_219], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf331, buf343, arg271_1, arg272_1, arg273_1, buf347, 1568, 256, grid=grid(1568), stream=stream0)
        del arg272_1
        del arg273_1
        buf348 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf347, (1568, 256), (256, 1), 0), reinterpret_tensor(arg274_1, (256, 1536), (1, 256), 0), out=buf348)
        del arg274_1
        buf349 = buf338; del buf338  # reuse
        buf350 = buf337; del buf337  # reuse
        # Source Nodes: [v_82], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf348, arg275_1, buf349, buf350, 1568, 768, grid=grid(1568), stream=stream0)
        buf352 = reinterpret_tensor(buf342, (8, 768, 196), (150528, 196, 1), 0); del buf342  # reuse
        # Source Nodes: [v_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf348, arg275_1, buf349, buf350, arg276_1, arg277_1, buf352, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg276_1
        del arg277_1
        buf353 = buf341; del buf341  # reuse
        # Source Nodes: [v_83], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (6144, 196), (196, 1), 0), reinterpret_tensor(arg278_1, (196, 196), (1, 196), 0), out=buf353)
        del arg278_1
        buf354 = reinterpret_tensor(buf352, (8, 196, 768), (150528, 768, 1), 0); del buf352  # reuse
        # Source Nodes: [x_223], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf348, arg275_1, buf353, arg279_1, buf354, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg275_1
        del arg279_1
        buf355 = reinterpret_tensor(buf347, (1568, 256), (256, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (1568, 768), (768, 1), 0), reinterpret_tensor(arg280_1, (768, 256), (1, 768), 0), out=buf355)
        del arg280_1
        buf356 = reinterpret_tensor(buf355, (8, 196, 256), (50176, 256, 1), 0); del buf355  # reuse
        buf360 = buf306; del buf306  # reuse
        # Source Nodes: [getattr_l__mod___blocks___28___norm, x_219, x_227], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf356, buf331, buf343, arg271_1, arg281_1, arg282_1, arg283_1, buf360, 1568, 256, grid=grid(1568), stream=stream0)
        del arg271_1
        del arg281_1
        del arg282_1
        del arg283_1
        del buf331
        buf361 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf360, (1568, 256), (256, 1), 0), reinterpret_tensor(arg284_1, (256, 1536), (1, 256), 0), out=buf361)
        del arg284_1
        buf362 = buf350; del buf350  # reuse
        buf363 = buf349; del buf349  # reuse
        # Source Nodes: [v_85], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf361, arg285_1, buf362, buf363, 1568, 768, grid=grid(1568), stream=stream0)
        buf365 = reinterpret_tensor(buf354, (8, 768, 196), (150528, 196, 1), 0); del buf354  # reuse
        # Source Nodes: [v_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf361, arg285_1, buf362, buf363, arg286_1, arg287_1, buf365, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg286_1
        del arg287_1
        buf366 = buf353; del buf353  # reuse
        # Source Nodes: [v_86], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (6144, 196), (196, 1), 0), reinterpret_tensor(arg288_1, (196, 196), (1, 196), 0), out=buf366)
        del arg288_1
        buf367 = reinterpret_tensor(buf365, (8, 196, 768), (150528, 768, 1), 0); del buf365  # reuse
        # Source Nodes: [x_231], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf361, arg285_1, buf366, arg289_1, buf367, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg285_1
        del arg289_1
        buf368 = reinterpret_tensor(buf360, (1568, 256), (256, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf367, (1568, 768), (768, 1), 0), reinterpret_tensor(arg290_1, (768, 256), (1, 768), 0), out=buf368)
        del arg290_1
        buf372 = reinterpret_tensor(buf343, (8, 196, 256), (50176, 256, 1), 0); del buf343  # reuse
        # Source Nodes: [getattr_l__mod___blocks___29___norm, x_235], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf356, buf368, arg291_1, arg292_1, arg293_1, buf372, 1568, 256, grid=grid(1568), stream=stream0)
        del arg292_1
        del arg293_1
        buf373 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (1568, 256), (256, 1), 0), reinterpret_tensor(arg294_1, (256, 1536), (1, 256), 0), out=buf373)
        del arg294_1
        buf374 = buf363; del buf363  # reuse
        buf375 = buf362; del buf362  # reuse
        # Source Nodes: [v_88], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf373, arg295_1, buf374, buf375, 1568, 768, grid=grid(1568), stream=stream0)
        buf377 = reinterpret_tensor(buf367, (8, 768, 196), (150528, 196, 1), 0); del buf367  # reuse
        # Source Nodes: [v_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf373, arg295_1, buf374, buf375, arg296_1, arg297_1, buf377, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg296_1
        del arg297_1
        buf378 = buf366; del buf366  # reuse
        # Source Nodes: [v_89], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (6144, 196), (196, 1), 0), reinterpret_tensor(arg298_1, (196, 196), (1, 196), 0), out=buf378)
        del arg298_1
        buf379 = reinterpret_tensor(buf377, (8, 196, 768), (150528, 768, 1), 0); del buf377  # reuse
        # Source Nodes: [x_239], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf373, arg295_1, buf378, arg299_1, buf379, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg295_1
        del arg299_1
        del buf373
        del buf378
        buf380 = reinterpret_tensor(buf372, (1568, 256), (256, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf379, (1568, 768), (768, 1), 0), reinterpret_tensor(arg300_1, (768, 256), (1, 768), 0), out=buf380)
        del arg300_1
        del buf379
        buf381 = reinterpret_tensor(buf380, (8, 196, 256), (50176, 256, 1), 0); del buf380  # reuse
        buf382 = buf375; del buf375  # reuse
        buf383 = buf374; del buf374  # reuse
        # Source Nodes: [x_235, x_244, x_246], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf381, buf356, buf368, arg291_1, arg301_1, buf382, buf383, 1568, 256, grid=grid(1568), stream=stream0)
        del arg291_1
        del arg301_1
        del buf356
        del buf368
        buf385 = empty_strided((8, 256, 2), (512, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_red_fused_mean_native_layer_norm_13.run(buf381, buf382, buf383, arg302_1, arg303_1, buf385, 4096, 98, grid=grid(4096), stream=stream0)
        del arg302_1
        del arg303_1
        del buf381
        del buf382
        del buf383
        buf386 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf387 = buf386; del buf386  # reuse
        # Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_14.run(buf387, buf385, 2048, 2, grid=grid(2048), stream=stream0)
        del buf385
        buf388 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_246, x_247, x_249], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
        extern_kernels.addmm(arg305_1, buf387, reinterpret_tensor(arg304_1, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf388)
        del arg304_1
        del arg305_1
        return (buf388, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmlp_s16_224', benchmark_compiled_module)
