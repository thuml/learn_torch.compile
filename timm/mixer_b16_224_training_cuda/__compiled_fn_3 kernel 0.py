
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


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnrztk2rmko3dhpusxqdwpejdm6pm73v2hll3xrjsuhqd7kvkpg.py
# Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm1 => clone, var_mean
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 6
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


# kernel path: /tmp/torchinductor_youkaichao/37/c37o3amnvm55lphjzipltohzm6eauovshxznxrgw52ke6t3ifg3b.py
# Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___blocks___0___norm1 => add, clone, rsqrt, var_mean
triton_per_fused_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
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
    tmp16 = 768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x3), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwr6b6kqqrysrzn3xuryxvh77llazkqnuzyrmrbx3rkp4trehllg.py
# Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm1 => add, clone, mul, rsqrt, sub, var_mean
triton_poi_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': []},
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
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrpvxx6ygjcbhil27jrwxshttch3yxb6il3wi3eenluagfbwyzd.py
# Source Nodes: [x_4], Original ATen: [aten._unsafe_view, aten.clone]
# x_4 => clone_1, view_1
triton_poi_fused__unsafe_view_clone_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((768*x1) + (150528*(y0 // 768)) + (y0 % 768)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 % 768), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 % 768), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x1 + (196*y0)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhhbxb4f2jckwmtrypfhq3popwugef3go34smntxpqgzshq4laq.py
# Source Nodes: [x_4, x_5, x_8], Original ATen: [aten.add, aten.gelu, aten.view]
# x_4 => add_2
# x_5 => add_3, erf, mul_2, mul_3, mul_4
# x_8 => view_3
triton_poi_fused_add_gelu_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmto6ijdlpahb42vi4htdsyveybpys2otmvkp3gtz6q6zo4vazoa.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm2 => clone_4, var_mean_1
# x_10 => add_4
triton_red_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 6
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsulgn66sp76wxhtw532flcerc5hy2eyaudr4ojzjmni4ko6dg2.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm2 => add_5, clone_4, mul_5, rsqrt_1, sub_1, var_mean_1
# x_10 => add_4
triton_poi_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_6', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 768.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wj6grge67zxau6smpgtoereukp7je3li7gle6alw4gidjv3fcc.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_11], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_l__mod___blocks___0___norm2 => add_6, mul_6
# x_11 => view_5
triton_poi_fused_native_layer_norm_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3eyza3mh5a6bkbcgi3qrt36mvacpszrzom4gc5qpe3f7dellqy.py
# Source Nodes: [x_12, x_15], Original ATen: [aten.gelu, aten.view]
# x_12 => add_7, erf_1, mul_7, mul_8, mul_9
# x_15 => view_7
triton_poi_fused_gelu_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
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


# kernel path: /tmp/torchinductor_youkaichao/oe/coedtmrwcwagu5svfzqg3c2ggw74t33d3zwlhv42egbdtwep32o5.py
# Source Nodes: [x_10, x_17], Original ATen: [aten.add]
# x_10 => add_4
# x_17 => add_8
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/crybjgimcoiveqqosdxqmtfwobayrjjogw62bgmxsoixy5fa3axt.py
# Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm1 => clone_7, var_mean_2
triton_red_fused_native_layer_norm_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 196
    x2 = (xindex // 1176)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (196*x0) + (1176*x2)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (196*x0) + (1176*x2)), tmp3, xmask)
    tl.store(out_ptr2 + (x1 + (196*x0) + (1176*x2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3dkxckj5t6nltvcgookljayzl2catusnzkgmtqfsgdqv7irtrn.py
# Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm1 => add_9, clone_7, mul_10, rsqrt_2, sub_2, var_mean_2
triton_poi_fused_native_layer_norm_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dzvpzfyqv2kilfrfom4zqtezmjuuj6htolaqyeweuur6vqdjxe.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm2 => clone_11, var_mean_3
# x_24 => add_13
triton_red_fused_add_native_layer_norm_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jz/cjzxaspdczdgndttmzrx2e4br7w2enlg44mclxafqcqfts235iro.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24, x_25], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_l__mod___blocks___1___norm2 => add_14, add_15, clone_11, mul_15, mul_16, rsqrt_3, sub_3, var_mean_3
# x_24 => add_13
# x_25 => view_13
triton_poi_fused_add_native_layer_norm_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp13, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (768*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cilt6lbo3hbgrv5vqnnx3ujq7obmzjvc2kfnk6vrqkwogqx7hpvp.py
# Source Nodes: [x_24, x_31], Original ATen: [aten.add]
# x_24 => add_13
# x_31 => add_17
triton_poi_fused_add_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
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
    tmp0 = tl.load(in_out_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (196*x2) + (150528*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pm/cpmlnzzllbefjwsoiyy64h7hnmqba6rtb2hkmgexyqogtjm7m3xn.py
# Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___2___norm1 => clone_14, var_mean_4
triton_red_fused_native_layer_norm_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7stpg2btvypieyhi7we3joku2iqg2iogxnstycnpqqfxlrndoy.py
# Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___2___norm2 => add_23, clone_18, mul_25, rsqrt_5, sub_5, var_mean_5
# x_38 => add_22
triton_poi_fused_add_native_layer_norm_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfkffmizqurhn6fcoc7fob63ijraioir7o3flnlcj7hnuoq5d4me.py
# Source Nodes: [x_38, x_45], Original ATen: [aten.add]
# x_38 => add_22
# x_45 => add_26
triton_poi_fused_add_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c75ga2iigppfbn5ykncmopg2cia7n6vtj7mutf2bulmpctzbypzx.py
# Source Nodes: [x_66, x_73], Original ATen: [aten.add]
# x_66 => add_40
# x_73 => add_44
triton_poi_fused_add_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6faktvax4tvkirsgsnf2yxu2ko6wilghqbjun43eifi4r3nzs3.py
# Source Nodes: [x_174, x_175], Original ATen: [aten.mean, aten.native_layer_norm]
# x_174 => add_109, mul_121
# x_175 => mean
triton_red_fused_mean_native_layer_norm_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_layer_norm_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (75264*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cjetfwgxwmlmucjkbj4v6nwuan3al5yf6zpmdmbmycl6ke5bmawz.py
# Source Nodes: [x_174, x_175], Original ATen: [aten.mean, aten.native_layer_norm]
# x_174 => add_109, mul_121
# x_175 => mean
triton_per_fused_mean_native_layer_norm_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (1536*x1)), rmask, other=0.0)
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (384, 196), (196, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (196, 384), (384, 1))
    assert_size_stride(primals_8, (196, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (3072, 768), (768, 1))
    assert_size_stride(primals_12, (3072, ), (1, ))
    assert_size_stride(primals_13, (768, 3072), (3072, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (384, 196), (196, 1))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (196, 384), (384, 1))
    assert_size_stride(primals_20, (196, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (3072, 768), (768, 1))
    assert_size_stride(primals_24, (3072, ), (1, ))
    assert_size_stride(primals_25, (768, 3072), (3072, 1))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (384, 196), (196, 1))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_31, (196, 384), (384, 1))
    assert_size_stride(primals_32, (196, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (3072, 768), (768, 1))
    assert_size_stride(primals_36, (3072, ), (1, ))
    assert_size_stride(primals_37, (768, 3072), (3072, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (384, 196), (196, 1))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (196, 384), (384, 1))
    assert_size_stride(primals_44, (196, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (3072, 768), (768, 1))
    assert_size_stride(primals_48, (3072, ), (1, ))
    assert_size_stride(primals_49, (768, 3072), (3072, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (384, 196), (196, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (196, 384), (384, 1))
    assert_size_stride(primals_56, (196, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (3072, 768), (768, 1))
    assert_size_stride(primals_60, (3072, ), (1, ))
    assert_size_stride(primals_61, (768, 3072), (3072, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (384, 196), (196, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (196, 384), (384, 1))
    assert_size_stride(primals_68, (196, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (3072, 768), (768, 1))
    assert_size_stride(primals_72, (3072, ), (1, ))
    assert_size_stride(primals_73, (768, 3072), (3072, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (384, 196), (196, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (196, 384), (384, 1))
    assert_size_stride(primals_80, (196, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (3072, 768), (768, 1))
    assert_size_stride(primals_84, (3072, ), (1, ))
    assert_size_stride(primals_85, (768, 3072), (3072, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (384, 196), (196, 1))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (196, 384), (384, 1))
    assert_size_stride(primals_92, (196, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (3072, 768), (768, 1))
    assert_size_stride(primals_96, (3072, ), (1, ))
    assert_size_stride(primals_97, (768, 3072), (3072, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (384, 196), (196, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (196, 384), (384, 1))
    assert_size_stride(primals_104, (196, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (3072, 768), (768, 1))
    assert_size_stride(primals_108, (3072, ), (1, ))
    assert_size_stride(primals_109, (768, 3072), (3072, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (384, 196), (196, 1))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (196, 384), (384, 1))
    assert_size_stride(primals_116, (196, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (3072, 768), (768, 1))
    assert_size_stride(primals_120, (3072, ), (1, ))
    assert_size_stride(primals_121, (768, 3072), (3072, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (384, 196), (196, 1))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (196, 384), (384, 1))
    assert_size_stride(primals_128, (196, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (3072, 768), (768, 1))
    assert_size_stride(primals_132, (3072, ), (1, ))
    assert_size_stride(primals_133, (768, 3072), (3072, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (384, 196), (196, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (196, 384), (384, 1))
    assert_size_stride(primals_140, (196, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (3072, 768), (768, 1))
    assert_size_stride(primals_144, (3072, ), (1, ))
    assert_size_stride(primals_145, (768, 3072), (3072, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (1000, 768), (768, 1))
    assert_size_stride(primals_150, (1000, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_151, primals_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        buf1 = empty_strided((8, 196, 1, 6), (1176, 1, 9408, 196), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 6), (1176, 1, 9408, 196), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 6), (1176, 1, 9408, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, primals_2, buf1, buf2, buf3, 9408, 128, grid=grid(9408), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf312 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf1, buf2, buf3, buf4, buf5, buf312, 1568, 6, grid=grid(1568), stream=stream0)
        buf7 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_2.run(buf0, primals_2, buf4, buf5, buf7, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf8 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf7, primals_3, primals_4, buf8, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_4
        buf9 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, reinterpret_tensor(primals_5, (196, 384), (1, 196), 0), out=buf9)
        buf10 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_5, x_8], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf9, primals_6, buf10, 2359296, grid=grid(2359296), stream=stream0)
        buf11 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf10, reinterpret_tensor(primals_7, (384, 196), (1, 384), 0), out=buf11)
        buf12 = buf3; del buf3  # reuse
        buf13 = buf2; del buf2  # reuse
        buf14 = buf1; del buf1  # reuse
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf0, primals_2, buf11, primals_8, buf12, buf13, buf14, 9408, 128, grid=grid(9408), stream=stream0)
        buf15 = buf5; del buf5  # reuse
        buf16 = buf4; del buf4  # reuse
        buf311 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf12, buf13, buf14, buf15, buf16, buf311, 1568, 6, grid=grid(1568), stream=stream0)
        buf18 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_6.run(buf0, primals_2, buf11, primals_8, buf15, buf16, buf18, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf19 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_11], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf18, primals_9, primals_10, buf19, 1204224, grid=grid(1204224), stream=stream0)
        del primals_10
        buf20 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, buf19, reinterpret_tensor(primals_11, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf20)
        del primals_12
        buf21 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_15], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf20, buf21, 4816896, grid=grid(4816896), stream=stream0)
        buf22 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf21, reinterpret_tensor(primals_13, (3072, 768), (1, 3072), 0), out=buf22)
        buf23 = reinterpret_tensor(buf0, (8, 196, 768), (150528, 1, 196), 0); del buf0  # reuse
        # Source Nodes: [x_10, x_17], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf23, primals_2, buf11, primals_8, buf22, primals_14, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_14
        del primals_2
        del primals_8
        buf24 = buf14; del buf14  # reuse
        buf25 = buf13; del buf13  # reuse
        buf26 = buf12; del buf12  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf23, buf24, buf25, buf26, 9408, 128, grid=grid(9408), stream=stream0)
        buf27 = buf16; del buf16  # reuse
        buf28 = buf15; del buf15  # reuse
        buf310 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf24, buf25, buf26, buf27, buf28, buf310, 1568, 6, grid=grid(1568), stream=stream0)
        buf30 = reinterpret_tensor(buf22, (8, 196, 768), (150528, 768, 1), 0); del buf22  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf23, buf27, buf28, buf30, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf31 = buf11; del buf11  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf30, primals_15, primals_16, buf31, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_16
        buf32 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, reinterpret_tensor(primals_17, (196, 384), (1, 196), 0), out=buf32)
        buf33 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18, x_19, x_22], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf32, primals_18, buf33, 2359296, grid=grid(2359296), stream=stream0)
        buf34 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_19, (384, 196), (1, 384), 0), out=buf34)
        buf35 = buf26; del buf26  # reuse
        buf36 = buf25; del buf25  # reuse
        buf37 = buf24; del buf24  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf23, buf34, primals_20, buf35, buf36, buf37, 9408, 128, grid=grid(9408), stream=stream0)
        buf38 = buf28; del buf28  # reuse
        buf39 = buf27; del buf27  # reuse
        buf309 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf35, buf36, buf37, buf38, buf39, buf309, 1568, 6, grid=grid(1568), stream=stream0)
        buf41 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf42 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24, x_25], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf23, buf34, primals_20, buf38, buf39, primals_21, primals_22, buf41, buf42, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_22
        buf43 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, buf42, reinterpret_tensor(primals_23, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf43)
        del primals_24
        buf44 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26, x_29], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf43, buf44, 4816896, grid=grid(4816896), stream=stream0)
        buf45 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_25, (3072, 768), (1, 3072), 0), out=buf45)
        buf46 = buf23; del buf23  # reuse
        # Source Nodes: [x_24, x_31], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf46, buf34, primals_20, buf45, primals_26, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_20
        del primals_26
        buf47 = buf37; del buf37  # reuse
        buf48 = buf36; del buf36  # reuse
        buf49 = buf35; del buf35  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf46, buf47, buf48, buf49, 9408, 128, grid=grid(9408), stream=stream0)
        buf50 = buf39; del buf39  # reuse
        buf51 = buf38; del buf38  # reuse
        buf308 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf47, buf48, buf49, buf50, buf51, buf308, 1568, 6, grid=grid(1568), stream=stream0)
        buf53 = reinterpret_tensor(buf45, (8, 196, 768), (150528, 768, 1), 0); del buf45  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf46, buf50, buf51, buf53, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf54 = buf34; del buf34  # reuse
        # Source Nodes: [x_32], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf53, primals_27, primals_28, buf54, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_28
        buf55 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.mm]
        extern_kernels.mm(buf54, reinterpret_tensor(primals_29, (196, 384), (1, 196), 0), out=buf55)
        buf56 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32, x_33, x_36], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf55, primals_30, buf56, 2359296, grid=grid(2359296), stream=stream0)
        buf57 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(primals_31, (384, 196), (1, 384), 0), out=buf57)
        buf58 = buf49; del buf49  # reuse
        buf59 = buf48; del buf48  # reuse
        buf60 = buf47; del buf47  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf46, buf57, primals_32, buf58, buf59, buf60, 9408, 128, grid=grid(9408), stream=stream0)
        buf61 = buf51; del buf51  # reuse
        buf62 = buf50; del buf50  # reuse
        buf307 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf58, buf59, buf60, buf61, buf62, buf307, 1568, 6, grid=grid(1568), stream=stream0)
        buf64 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf46, buf57, primals_32, buf61, buf62, buf64, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf65 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_39], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf64, primals_33, primals_34, buf65, 1204224, grid=grid(1204224), stream=stream0)
        del primals_34
        buf66 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_36, buf65, reinterpret_tensor(primals_35, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf66)
        del primals_36
        buf67 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40, x_43], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf66, buf67, 4816896, grid=grid(4816896), stream=stream0)
        buf68 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf67, reinterpret_tensor(primals_37, (3072, 768), (1, 3072), 0), out=buf68)
        buf69 = buf46; del buf46  # reuse
        # Source Nodes: [x_38, x_45], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf69, buf57, primals_32, buf68, primals_38, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_32
        del primals_38
        buf70 = buf60; del buf60  # reuse
        buf71 = buf59; del buf59  # reuse
        buf72 = buf58; del buf58  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf69, buf70, buf71, buf72, 9408, 128, grid=grid(9408), stream=stream0)
        buf73 = buf62; del buf62  # reuse
        buf74 = buf61; del buf61  # reuse
        buf306 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf70, buf71, buf72, buf73, buf74, buf306, 1568, 6, grid=grid(1568), stream=stream0)
        buf76 = reinterpret_tensor(buf68, (8, 196, 768), (150528, 768, 1), 0); del buf68  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf69, buf73, buf74, buf76, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf77 = buf57; del buf57  # reuse
        # Source Nodes: [x_46], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf76, primals_39, primals_40, buf77, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_40
        buf78 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.mm]
        extern_kernels.mm(buf77, reinterpret_tensor(primals_41, (196, 384), (1, 196), 0), out=buf78)
        buf79 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46, x_47, x_50], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf78, primals_42, buf79, 2359296, grid=grid(2359296), stream=stream0)
        buf80 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf79, reinterpret_tensor(primals_43, (384, 196), (1, 384), 0), out=buf80)
        buf81 = buf72; del buf72  # reuse
        buf82 = buf71; del buf71  # reuse
        buf83 = buf70; del buf70  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf69, buf80, primals_44, buf81, buf82, buf83, 9408, 128, grid=grid(9408), stream=stream0)
        buf84 = buf74; del buf74  # reuse
        buf85 = buf73; del buf73  # reuse
        buf305 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf81, buf82, buf83, buf84, buf85, buf305, 1568, 6, grid=grid(1568), stream=stream0)
        buf87 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf88 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52, x_53], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf69, buf80, primals_44, buf84, buf85, primals_45, primals_46, buf87, buf88, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_46
        buf89 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, buf88, reinterpret_tensor(primals_47, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf89)
        del primals_48
        buf90 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54, x_57], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf89, buf90, 4816896, grid=grid(4816896), stream=stream0)
        buf91 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf90, reinterpret_tensor(primals_49, (3072, 768), (1, 3072), 0), out=buf91)
        buf92 = buf69; del buf69  # reuse
        # Source Nodes: [x_52, x_59], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf92, buf80, primals_44, buf91, primals_50, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_44
        del primals_50
        buf93 = buf83; del buf83  # reuse
        buf94 = buf82; del buf82  # reuse
        buf95 = buf81; del buf81  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf92, buf93, buf94, buf95, 9408, 128, grid=grid(9408), stream=stream0)
        buf96 = buf85; del buf85  # reuse
        buf97 = buf84; del buf84  # reuse
        buf304 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf93, buf94, buf95, buf96, buf97, buf304, 1568, 6, grid=grid(1568), stream=stream0)
        buf99 = reinterpret_tensor(buf91, (8, 196, 768), (150528, 768, 1), 0); del buf91  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf92, buf96, buf97, buf99, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf100 = buf80; del buf80  # reuse
        # Source Nodes: [x_60], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf99, primals_51, primals_52, buf100, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_52
        buf101 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.mm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_53, (196, 384), (1, 196), 0), out=buf101)
        buf102 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60, x_61, x_64], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf101, primals_54, buf102, 2359296, grid=grid(2359296), stream=stream0)
        buf103 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf102, reinterpret_tensor(primals_55, (384, 196), (1, 384), 0), out=buf103)
        buf104 = buf95; del buf95  # reuse
        buf105 = buf94; del buf94  # reuse
        buf106 = buf93; del buf93  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf92, buf103, primals_56, buf104, buf105, buf106, 9408, 128, grid=grid(9408), stream=stream0)
        buf107 = buf97; del buf97  # reuse
        buf108 = buf96; del buf96  # reuse
        buf303 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf104, buf105, buf106, buf107, buf108, buf303, 1568, 6, grid=grid(1568), stream=stream0)
        buf110 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf92, buf103, primals_56, buf107, buf108, buf110, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf111 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_67], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf110, primals_57, primals_58, buf111, 1204224, grid=grid(1204224), stream=stream0)
        del primals_58
        buf112 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, buf111, reinterpret_tensor(primals_59, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf112)
        del primals_60
        buf113 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68, x_71], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf112, buf113, 4816896, grid=grid(4816896), stream=stream0)
        buf114 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf113, reinterpret_tensor(primals_61, (3072, 768), (1, 3072), 0), out=buf114)
        buf115 = reinterpret_tensor(buf103, (8, 196, 768), (150528, 1, 196), 0); del buf103  # reuse
        # Source Nodes: [x_66, x_73], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf115, buf92, primals_56, buf114, primals_62, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_56
        del primals_62
        buf116 = buf106; del buf106  # reuse
        buf117 = buf105; del buf105  # reuse
        buf118 = buf104; del buf104  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf115, buf116, buf117, buf118, 9408, 128, grid=grid(9408), stream=stream0)
        buf119 = buf108; del buf108  # reuse
        buf120 = buf107; del buf107  # reuse
        buf302 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf116, buf117, buf118, buf119, buf120, buf302, 1568, 6, grid=grid(1568), stream=stream0)
        buf122 = reinterpret_tensor(buf92, (8, 196, 768), (150528, 768, 1), 0); del buf92  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf115, buf119, buf120, buf122, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf123 = reinterpret_tensor(buf114, (6144, 196), (196, 1), 0); del buf114  # reuse
        # Source Nodes: [x_74], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf122, primals_63, primals_64, buf123, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_64
        buf124 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.mm]
        extern_kernels.mm(buf123, reinterpret_tensor(primals_65, (196, 384), (1, 196), 0), out=buf124)
        buf125 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_75, x_78], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf124, primals_66, buf125, 2359296, grid=grid(2359296), stream=stream0)
        buf126 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf125, reinterpret_tensor(primals_67, (384, 196), (1, 384), 0), out=buf126)
        buf127 = buf118; del buf118  # reuse
        buf128 = buf117; del buf117  # reuse
        buf129 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf115, buf126, primals_68, buf127, buf128, buf129, 9408, 128, grid=grid(9408), stream=stream0)
        buf130 = buf120; del buf120  # reuse
        buf131 = buf119; del buf119  # reuse
        buf301 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf127, buf128, buf129, buf130, buf131, buf301, 1568, 6, grid=grid(1568), stream=stream0)
        buf133 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf134 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80, x_81], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf115, buf126, primals_68, buf130, buf131, primals_69, primals_70, buf133, buf134, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_70
        buf135 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_72, buf134, reinterpret_tensor(primals_71, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf135)
        del primals_72
        buf136 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82, x_85], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf135, buf136, 4816896, grid=grid(4816896), stream=stream0)
        buf137 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf136, reinterpret_tensor(primals_73, (3072, 768), (1, 3072), 0), out=buf137)
        buf138 = buf115; del buf115  # reuse
        # Source Nodes: [x_80, x_87], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf138, buf126, primals_68, buf137, primals_74, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_68
        del primals_74
        buf139 = buf129; del buf129  # reuse
        buf140 = buf128; del buf128  # reuse
        buf141 = buf127; del buf127  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf138, buf139, buf140, buf141, 9408, 128, grid=grid(9408), stream=stream0)
        buf142 = buf131; del buf131  # reuse
        buf143 = buf130; del buf130  # reuse
        buf300 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf139, buf140, buf141, buf142, buf143, buf300, 1568, 6, grid=grid(1568), stream=stream0)
        buf145 = reinterpret_tensor(buf137, (8, 196, 768), (150528, 768, 1), 0); del buf137  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf138, buf142, buf143, buf145, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf146 = buf126; del buf126  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf145, primals_75, primals_76, buf146, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_76
        buf147 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.mm]
        extern_kernels.mm(buf146, reinterpret_tensor(primals_77, (196, 384), (1, 196), 0), out=buf147)
        buf148 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88, x_89, x_92], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf147, primals_78, buf148, 2359296, grid=grid(2359296), stream=stream0)
        buf149 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf148, reinterpret_tensor(primals_79, (384, 196), (1, 384), 0), out=buf149)
        buf150 = buf141; del buf141  # reuse
        buf151 = buf140; del buf140  # reuse
        buf152 = buf139; del buf139  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf138, buf149, primals_80, buf150, buf151, buf152, 9408, 128, grid=grid(9408), stream=stream0)
        buf153 = buf143; del buf143  # reuse
        buf154 = buf142; del buf142  # reuse
        buf299 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf150, buf151, buf152, buf153, buf154, buf299, 1568, 6, grid=grid(1568), stream=stream0)
        buf156 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf138, buf149, primals_80, buf153, buf154, buf156, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf157 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_95], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf156, primals_81, primals_82, buf157, 1204224, grid=grid(1204224), stream=stream0)
        del primals_82
        buf158 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_84, buf157, reinterpret_tensor(primals_83, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf158)
        del primals_84
        buf159 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96, x_99], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf158, buf159, 4816896, grid=grid(4816896), stream=stream0)
        buf160 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf159, reinterpret_tensor(primals_85, (3072, 768), (1, 3072), 0), out=buf160)
        buf161 = buf138; del buf138  # reuse
        # Source Nodes: [x_101, x_94], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf161, buf149, primals_80, buf160, primals_86, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_80
        del primals_86
        buf162 = buf152; del buf152  # reuse
        buf163 = buf151; del buf151  # reuse
        buf164 = buf150; del buf150  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf161, buf162, buf163, buf164, 9408, 128, grid=grid(9408), stream=stream0)
        buf165 = buf154; del buf154  # reuse
        buf166 = buf153; del buf153  # reuse
        buf298 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf162, buf163, buf164, buf165, buf166, buf298, 1568, 6, grid=grid(1568), stream=stream0)
        buf168 = reinterpret_tensor(buf160, (8, 196, 768), (150528, 768, 1), 0); del buf160  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf161, buf165, buf166, buf168, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf169 = buf149; del buf149  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf168, primals_87, primals_88, buf169, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_88
        buf170 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten.mm]
        extern_kernels.mm(buf169, reinterpret_tensor(primals_89, (196, 384), (1, 196), 0), out=buf170)
        buf171 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_103, x_106], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf170, primals_90, buf171, 2359296, grid=grid(2359296), stream=stream0)
        buf172 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, reinterpret_tensor(primals_91, (384, 196), (1, 384), 0), out=buf172)
        buf173 = buf164; del buf164  # reuse
        buf174 = buf163; del buf163  # reuse
        buf175 = buf162; del buf162  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf161, buf172, primals_92, buf173, buf174, buf175, 9408, 128, grid=grid(9408), stream=stream0)
        buf176 = buf166; del buf166  # reuse
        buf177 = buf165; del buf165  # reuse
        buf297 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf173, buf174, buf175, buf176, buf177, buf297, 1568, 6, grid=grid(1568), stream=stream0)
        buf179 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf180 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108, x_109], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf161, buf172, primals_92, buf176, buf177, primals_93, primals_94, buf179, buf180, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_94
        buf181 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf180, reinterpret_tensor(primals_95, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf181)
        del primals_96
        buf182 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110, x_113], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf181, buf182, 4816896, grid=grid(4816896), stream=stream0)
        buf183 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf182, reinterpret_tensor(primals_97, (3072, 768), (1, 3072), 0), out=buf183)
        buf184 = buf161; del buf161  # reuse
        # Source Nodes: [x_108, x_115], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf184, buf172, primals_92, buf183, primals_98, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_92
        del primals_98
        buf185 = buf175; del buf175  # reuse
        buf186 = buf174; del buf174  # reuse
        buf187 = buf173; del buf173  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf184, buf185, buf186, buf187, 9408, 128, grid=grid(9408), stream=stream0)
        buf188 = buf177; del buf177  # reuse
        buf189 = buf176; del buf176  # reuse
        buf296 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf185, buf186, buf187, buf188, buf189, buf296, 1568, 6, grid=grid(1568), stream=stream0)
        buf191 = reinterpret_tensor(buf183, (8, 196, 768), (150528, 768, 1), 0); del buf183  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf184, buf188, buf189, buf191, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf192 = buf172; del buf172  # reuse
        # Source Nodes: [x_116], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf191, primals_99, primals_100, buf192, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_100
        buf193 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.mm]
        extern_kernels.mm(buf192, reinterpret_tensor(primals_101, (196, 384), (1, 196), 0), out=buf193)
        buf194 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116, x_117, x_120], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf193, primals_102, buf194, 2359296, grid=grid(2359296), stream=stream0)
        buf195 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf194, reinterpret_tensor(primals_103, (384, 196), (1, 384), 0), out=buf195)
        buf196 = buf187; del buf187  # reuse
        buf197 = buf186; del buf186  # reuse
        buf198 = buf185; del buf185  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf184, buf195, primals_104, buf196, buf197, buf198, 9408, 128, grid=grid(9408), stream=stream0)
        buf199 = buf189; del buf189  # reuse
        buf200 = buf188; del buf188  # reuse
        buf295 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf196, buf197, buf198, buf199, buf200, buf295, 1568, 6, grid=grid(1568), stream=stream0)
        buf202 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf184, buf195, primals_104, buf199, buf200, buf202, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf203 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_123], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf202, primals_105, primals_106, buf203, 1204224, grid=grid(1204224), stream=stream0)
        del primals_106
        buf204 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_108, buf203, reinterpret_tensor(primals_107, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf204)
        del primals_108
        buf205 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124, x_127], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf204, buf205, 4816896, grid=grid(4816896), stream=stream0)
        buf206 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf205, reinterpret_tensor(primals_109, (3072, 768), (1, 3072), 0), out=buf206)
        buf207 = buf184; del buf184  # reuse
        # Source Nodes: [x_122, x_129], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf207, buf195, primals_104, buf206, primals_110, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_104
        del primals_110
        buf208 = buf198; del buf198  # reuse
        buf209 = buf197; del buf197  # reuse
        buf210 = buf196; del buf196  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf207, buf208, buf209, buf210, 9408, 128, grid=grid(9408), stream=stream0)
        buf211 = buf200; del buf200  # reuse
        buf212 = buf199; del buf199  # reuse
        buf294 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf208, buf209, buf210, buf211, buf212, buf294, 1568, 6, grid=grid(1568), stream=stream0)
        buf214 = reinterpret_tensor(buf206, (8, 196, 768), (150528, 768, 1), 0); del buf206  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf207, buf211, buf212, buf214, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf215 = buf195; del buf195  # reuse
        # Source Nodes: [x_130], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf214, primals_111, primals_112, buf215, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_112
        buf216 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten.mm]
        extern_kernels.mm(buf215, reinterpret_tensor(primals_113, (196, 384), (1, 196), 0), out=buf216)
        buf217 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130, x_131, x_134], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf216, primals_114, buf217, 2359296, grid=grid(2359296), stream=stream0)
        buf218 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf217, reinterpret_tensor(primals_115, (384, 196), (1, 384), 0), out=buf218)
        buf219 = buf210; del buf210  # reuse
        buf220 = buf209; del buf209  # reuse
        buf221 = buf208; del buf208  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf207, buf218, primals_116, buf219, buf220, buf221, 9408, 128, grid=grid(9408), stream=stream0)
        buf222 = buf212; del buf212  # reuse
        buf223 = buf211; del buf211  # reuse
        buf293 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf219, buf220, buf221, buf222, buf223, buf293, 1568, 6, grid=grid(1568), stream=stream0)
        buf225 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf226 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136, x_137], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf207, buf218, primals_116, buf222, buf223, primals_117, primals_118, buf225, buf226, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_118
        buf227 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf226, reinterpret_tensor(primals_119, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf227)
        del primals_120
        buf228 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138, x_141], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf227, buf228, 4816896, grid=grid(4816896), stream=stream0)
        buf229 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf228, reinterpret_tensor(primals_121, (3072, 768), (1, 3072), 0), out=buf229)
        buf230 = buf207; del buf207  # reuse
        # Source Nodes: [x_136, x_143], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf230, buf218, primals_116, buf229, primals_122, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_116
        del primals_122
        buf231 = buf221; del buf221  # reuse
        buf232 = buf220; del buf220  # reuse
        buf233 = buf219; del buf219  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf230, buf231, buf232, buf233, 9408, 128, grid=grid(9408), stream=stream0)
        buf234 = buf223; del buf223  # reuse
        buf235 = buf222; del buf222  # reuse
        buf292 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf231, buf232, buf233, buf234, buf235, buf292, 1568, 6, grid=grid(1568), stream=stream0)
        buf237 = reinterpret_tensor(buf229, (8, 196, 768), (150528, 768, 1), 0); del buf229  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf230, buf234, buf235, buf237, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf238 = buf218; del buf218  # reuse
        # Source Nodes: [x_144], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf237, primals_123, primals_124, buf238, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_124
        buf239 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.mm]
        extern_kernels.mm(buf238, reinterpret_tensor(primals_125, (196, 384), (1, 196), 0), out=buf239)
        buf240 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144, x_145, x_148], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf239, primals_126, buf240, 2359296, grid=grid(2359296), stream=stream0)
        buf241 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf240, reinterpret_tensor(primals_127, (384, 196), (1, 384), 0), out=buf241)
        buf242 = buf233; del buf233  # reuse
        buf243 = buf232; del buf232  # reuse
        buf244 = buf231; del buf231  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf230, buf241, primals_128, buf242, buf243, buf244, 9408, 128, grid=grid(9408), stream=stream0)
        buf245 = buf235; del buf235  # reuse
        buf246 = buf234; del buf234  # reuse
        buf291 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf242, buf243, buf244, buf245, buf246, buf291, 1568, 6, grid=grid(1568), stream=stream0)
        buf248 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf230, buf241, primals_128, buf245, buf246, buf248, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf249 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_151], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf248, primals_129, primals_130, buf249, 1204224, grid=grid(1204224), stream=stream0)
        del primals_130
        buf250 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf249, reinterpret_tensor(primals_131, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf250)
        del primals_132
        buf251 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152, x_155], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf250, buf251, 4816896, grid=grid(4816896), stream=stream0)
        buf252 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf251, reinterpret_tensor(primals_133, (3072, 768), (1, 3072), 0), out=buf252)
        buf253 = buf230; del buf230  # reuse
        # Source Nodes: [x_150, x_157], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf253, buf241, primals_128, buf252, primals_134, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_128
        del primals_134
        buf254 = buf244; del buf244  # reuse
        buf255 = buf243; del buf243  # reuse
        buf256 = buf242; del buf242  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf253, buf254, buf255, buf256, 9408, 128, grid=grid(9408), stream=stream0)
        buf257 = buf246; del buf246  # reuse
        buf258 = buf245; del buf245  # reuse
        buf290 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf254, buf255, buf256, buf257, buf258, buf290, 1568, 6, grid=grid(1568), stream=stream0)
        buf260 = reinterpret_tensor(buf252, (8, 196, 768), (150528, 768, 1), 0); del buf252  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf253, buf257, buf258, buf260, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf261 = buf241; del buf241  # reuse
        # Source Nodes: [x_158], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf260, primals_135, primals_136, buf261, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_136
        buf262 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.mm]
        extern_kernels.mm(buf261, reinterpret_tensor(primals_137, (196, 384), (1, 196), 0), out=buf262)
        buf263 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158, x_159, x_162], Original ATen: [aten.add, aten.gelu, aten.view]
        triton_poi_fused_add_gelu_view_4.run(buf262, primals_138, buf263, 2359296, grid=grid(2359296), stream=stream0)
        buf264 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf263, reinterpret_tensor(primals_139, (384, 196), (1, 384), 0), out=buf264)
        buf265 = buf256; del buf256  # reuse
        buf266 = buf255; del buf255  # reuse
        buf267 = buf254; del buf254  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf253, buf264, primals_140, buf265, buf266, buf267, 9408, 128, grid=grid(9408), stream=stream0)
        buf268 = buf258; del buf258  # reuse
        buf269 = buf257; del buf257  # reuse
        buf289 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf265, buf266, buf267, buf268, buf269, buf289, 1568, 6, grid=grid(1568), stream=stream0)
        buf271 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf272 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164, x_165], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf253, buf264, primals_140, buf268, buf269, primals_141, primals_142, buf271, buf272, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_142
        buf273 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf272, reinterpret_tensor(primals_143, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf273)
        del primals_144
        buf274 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166, x_169], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf273, buf274, 4816896, grid=grid(4816896), stream=stream0)
        buf275 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf274, reinterpret_tensor(primals_145, (3072, 768), (1, 3072), 0), out=buf275)
        buf276 = buf253; del buf253  # reuse
        # Source Nodes: [x_164, x_172, x_174], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_14.run(buf276, buf264, primals_140, buf275, primals_146, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del buf264
        del primals_140
        del primals_146
        buf277 = buf267; del buf267  # reuse
        buf278 = buf266; del buf266  # reuse
        buf279 = buf265; del buf265  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf276, buf277, buf278, buf279, 9408, 128, grid=grid(9408), stream=stream0)
        buf280 = buf269; del buf269  # reuse
        buf281 = buf268; del buf268  # reuse
        buf288 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf277, buf278, buf279, buf280, buf281, buf288, 1568, 6, grid=grid(1568), stream=stream0)
        del buf277
        del buf278
        del buf279
        buf283 = reinterpret_tensor(buf275, (8, 196, 768), (150528, 768, 1), 0); del buf275  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf276, buf280, buf281, buf283, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del buf276
        del buf280
        del buf281
        buf284 = empty_strided((8, 768, 2), (1536, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174, x_175], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_red_fused_mean_native_layer_norm_19.run(buf283, primals_147, primals_148, buf284, 12288, 98, grid=grid(12288), stream=stream0)
        del primals_148
        buf285 = empty((8, 768), device='cuda', dtype=torch.float32)
        buf286 = buf285; del buf285  # reuse
        # Source Nodes: [x_174, x_175], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_20.run(buf286, buf284, 6144, 2, grid=grid(6144), stream=stream0)
        del buf284
        buf287 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf286, reinterpret_tensor(primals_149, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf287)
        del primals_150
        return (buf287, primals_1, primals_3, primals_6, primals_9, primals_15, primals_18, primals_21, primals_27, primals_30, primals_33, primals_39, primals_42, primals_45, primals_51, primals_54, primals_57, primals_63, primals_66, primals_69, primals_75, primals_78, primals_81, primals_87, primals_90, primals_93, primals_99, primals_102, primals_105, primals_111, primals_114, primals_117, primals_123, primals_126, primals_129, primals_135, primals_138, primals_141, primals_147, primals_151, buf7, buf8, buf9, buf10, buf18, buf19, buf20, buf21, buf30, buf31, buf32, buf33, buf41, buf42, buf43, buf44, buf53, buf54, buf55, buf56, buf64, buf65, buf66, buf67, buf76, buf77, buf78, buf79, buf87, buf88, buf89, buf90, buf99, buf100, buf101, buf102, buf110, buf111, buf112, buf113, buf122, buf123, buf124, buf125, buf133, buf134, buf135, buf136, buf145, buf146, buf147, buf148, buf156, buf157, buf158, buf159, buf168, buf169, buf170, buf171, buf179, buf180, buf181, buf182, buf191, buf192, buf193, buf194, buf202, buf203, buf204, buf205, buf214, buf215, buf216, buf217, buf225, buf226, buf227, buf228, buf237, buf238, buf239, buf240, buf248, buf249, buf250, buf251, buf260, buf261, buf262, buf263, buf271, buf272, buf273, buf274, buf283, buf286, reinterpret_tensor(primals_149, (1000, 768), (768, 1), 0), buf288, reinterpret_tensor(primals_145, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 768), (768, 1), 0), buf289, reinterpret_tensor(primals_139, (196, 384), (384, 1), 0), reinterpret_tensor(primals_137, (384, 196), (196, 1), 0), buf290, reinterpret_tensor(primals_133, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 768), (768, 1), 0), buf291, reinterpret_tensor(primals_127, (196, 384), (384, 1), 0), reinterpret_tensor(primals_125, (384, 196), (196, 1), 0), buf292, reinterpret_tensor(primals_121, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_119, (3072, 768), (768, 1), 0), buf293, reinterpret_tensor(primals_115, (196, 384), (384, 1), 0), reinterpret_tensor(primals_113, (384, 196), (196, 1), 0), buf294, reinterpret_tensor(primals_109, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 768), (768, 1), 0), buf295, reinterpret_tensor(primals_103, (196, 384), (384, 1), 0), reinterpret_tensor(primals_101, (384, 196), (196, 1), 0), buf296, reinterpret_tensor(primals_97, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 768), (768, 1), 0), buf297, reinterpret_tensor(primals_91, (196, 384), (384, 1), 0), reinterpret_tensor(primals_89, (384, 196), (196, 1), 0), buf298, reinterpret_tensor(primals_85, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 768), (768, 1), 0), buf299, reinterpret_tensor(primals_79, (196, 384), (384, 1), 0), reinterpret_tensor(primals_77, (384, 196), (196, 1), 0), buf300, reinterpret_tensor(primals_73, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 768), (768, 1), 0), buf301, reinterpret_tensor(primals_67, (196, 384), (384, 1), 0), reinterpret_tensor(primals_65, (384, 196), (196, 1), 0), buf302, reinterpret_tensor(primals_61, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 768), (768, 1), 0), buf303, reinterpret_tensor(primals_55, (196, 384), (384, 1), 0), reinterpret_tensor(primals_53, (384, 196), (196, 1), 0), buf304, reinterpret_tensor(primals_49, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_47, (3072, 768), (768, 1), 0), buf305, reinterpret_tensor(primals_43, (196, 384), (384, 1), 0), reinterpret_tensor(primals_41, (384, 196), (196, 1), 0), buf306, reinterpret_tensor(primals_37, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_35, (3072, 768), (768, 1), 0), buf307, reinterpret_tensor(primals_31, (196, 384), (384, 1), 0), reinterpret_tensor(primals_29, (384, 196), (196, 1), 0), buf308, reinterpret_tensor(primals_25, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_23, (3072, 768), (768, 1), 0), buf309, reinterpret_tensor(primals_19, (196, 384), (384, 1), 0), reinterpret_tensor(primals_17, (384, 196), (196, 1), 0), buf310, reinterpret_tensor(primals_13, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_11, (3072, 768), (768, 1), 0), buf311, reinterpret_tensor(primals_7, (196, 384), (384, 1), 0), reinterpret_tensor(primals_5, (384, 196), (196, 1), 0), buf312, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixer_b16_224', benchmark_compiled_module)
