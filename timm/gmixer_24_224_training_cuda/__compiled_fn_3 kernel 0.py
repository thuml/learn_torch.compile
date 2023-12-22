
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


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhf6sw4bnyedq3pyoymvvolcoc426gqovyjx5hy6kdex32nnyhd.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 3
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


# kernel path: /tmp/torchinductor_youkaichao/j2/cj26ds4uyne2cadnkcetgkp6l7nhqeufb3kvrd4oavoqnv4nygxg.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
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
    tmp16 = 384.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x3), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyz34p5okeeywnuvmxs3ceildl7q3yitfvm2cutuq3dbtmcctdc.py
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
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 384.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4ohok2tegjtfgb2ykb2jl4hbwr3cn5mdnijveaitd6qejz5as3.py
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
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((384*x1) + (75264*(y0 // 384)) + (y0 % 384)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 % 384), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 % 384), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x1 + (196*y0)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwyc7s7tlbp4w2uyqwhma6n4njb6gsm4slb7vbwro5qe3w5lyh3.py
# Source Nodes: [getattr_l__mod___blocks___0___mlp_tokens_act, x_5, x_8], Original ATen: [aten.mul, aten.silu, aten.view]
# getattr_l__mod___blocks___0___mlp_tokens_act => mul_2, sigmoid
# x_5 => mul_3
# x_8 => view_3
triton_poi_fused_mul_silu_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (192 + x0 + (384*x1)), None)
    tmp4 = tl.load(in_ptr1 + (192 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlstwg4sorseeh3yzuatd27judxw2llyprnhztnfrcx544z6y5r.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm2 => clone_4, var_mean_1
# x_10 => add_3
triton_red_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 3
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


# kernel path: /tmp/torchinductor_youkaichao/3o/c3onu35ria5cv5kjhxtksyb3fgo6rigxpymwr4vtdhypl772a3gp.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm2 => add_4, clone_4, mul_4, rsqrt_1, sub_1, var_mean_1
# x_10 => add_3
triton_poi_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
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
    tmp10 = 384.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfx3v7xysehow55dksmt4sjxbjqh4bwx652gjnvpeuptrtdn64av.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_11], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_l__mod___blocks___0___norm2 => add_5, mul_5
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4gqzowf4f4slrbgfsqbgwoddha7r5ye7xjpenc4pkg3pdyh7sr.py
# Source Nodes: [getattr_l__mod___blocks___0___mlp_channels_act, x_12, x_15], Original ATen: [aten.mul, aten.silu, aten.view]
# getattr_l__mod___blocks___0___mlp_channels_act => mul_6, sigmoid_1
# x_12 => mul_7
# x_15 => view_7
triton_poi_fused_mul_silu_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*x1)), None)
    tmp1 = tl.load(in_ptr0 + (768 + x0 + (1536*x1)), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnpiiauxi62xm7skoznekf2j7fsqaeshgplnzh27ijjuy2fvizx.py
# Source Nodes: [x_10, x_17], Original ATen: [aten.add]
# x_10 => add_3
# x_17 => add_6
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cec55qush4eoxef5c7cgdodriprh54isatzjt6xxrzfgy2jhy3y7.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (75264*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (196*x0) + (588*x2)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (196*x0) + (588*x2)), tmp3, xmask)
    tl.store(out_ptr2 + (x1 + (196*x0) + (588*x2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36ultc4rse2yhcgx4ensjveazayb2zh5rloe7e4kk5xriwsarpx.py
# Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm1 => add_7, clone_7, mul_8, rsqrt_2, sub_2, var_mean_2
triton_poi_fused_native_layer_norm_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/csefgy6lqrkfvp2st2hsqa5ea6z3su4dqngavz3kgntnm5h6pfo3.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm2 => clone_11, var_mean_3
# x_24 => add_10
triton_red_fused_add_native_layer_norm_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
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


# kernel path: /tmp/torchinductor_youkaichao/w2/cw25jbca6dqdiianra2r7cuyica4wq5d7tr4wmncm7wooih5meok.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24, x_25], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_l__mod___blocks___1___norm2 => add_11, add_12, clone_11, mul_12, mul_13, rsqrt_3, sub_3, var_mean_3
# x_24 => add_10
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
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp13, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (384*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtaa6mxqbqk72mwq5akmxiq7jdthp4bpsvpuszpvpyeagzfidia.py
# Source Nodes: [x_24, x_31], Original ATen: [aten.add]
# x_24 => add_10
# x_31 => add_13
triton_poi_fused_add_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_out_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqjioidmaywuo6eo7oumgj4s4v3ahun2q3kp75j7po4qbotw4di.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
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


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrcxyjsiagj5hzjhs3442z2wklup3lpyn6neoaenjv74tiyecv2.py
# Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___2___norm2 => add_18, clone_18, mul_20, rsqrt_5, sub_5, var_mean_5
# x_38 => add_17
triton_poi_fused_add_native_layer_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 384)
    y0 = yindex % 384
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctq336q576giuzsqlltnyym7f24dksfwup5nwsar7ezulibykhpi.py
# Source Nodes: [x_38, x_45], Original ATen: [aten.add]
# x_38 => add_17
# x_45 => add_20
triton_poi_fused_add_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cglpdmpfrgyifiz67iba436chl5fjfgutgzuzfqam5557jctkel6.py
# Source Nodes: [x_66, x_73], Original ATen: [aten.add]
# x_66 => add_31
# x_73 => add_34
triton_poi_fused_add_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwkxfutivvcwlftx6rywskf5hdvyighaumzfqtcreockmyrttzp.py
# Source Nodes: [x_342, x_343], Original ATen: [aten.mean, aten.native_layer_norm]
# x_342 => add_169, mul_193
# x_343 => mean
triton_red_fused_mean_native_layer_norm_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_layer_norm_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (37632*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6tozodex2wzze7htpn3o6kzwcr2nz7qn24sygscda7dmklsejg.py
# Source Nodes: [x_342, x_343], Original ATen: [aten.mean, aten.native_layer_norm]
# x_342 => add_169, mul_193
# x_343 => mean
triton_per_fused_mean_native_layer_norm_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (768*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vm/cvmfg2fwzqin2apaeqbd5sm4lm2lcepyb3f6d5hxhxrhmdbk26hm.py
# Source Nodes: [x_4], Original ATen: [aten.add]
# x_4 => add_2
triton_poi_fused_add_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295 = args
    args.clear()
    assert_size_stride(primals_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_2, (384, ), (1, ))
    assert_size_stride(primals_3, (384, ), (1, ))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_5, (384, 196), (196, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (196, 192), (192, 1))
    assert_size_stride(primals_8, (196, ), (1, ))
    assert_size_stride(primals_9, (384, ), (1, ))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_11, (1536, 384), (384, 1))
    assert_size_stride(primals_12, (1536, ), (1, ))
    assert_size_stride(primals_13, (384, 768), (768, 1))
    assert_size_stride(primals_14, (384, ), (1, ))
    assert_size_stride(primals_15, (384, ), (1, ))
    assert_size_stride(primals_16, (384, ), (1, ))
    assert_size_stride(primals_17, (384, 196), (196, 1))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (196, 192), (192, 1))
    assert_size_stride(primals_20, (196, ), (1, ))
    assert_size_stride(primals_21, (384, ), (1, ))
    assert_size_stride(primals_22, (384, ), (1, ))
    assert_size_stride(primals_23, (1536, 384), (384, 1))
    assert_size_stride(primals_24, (1536, ), (1, ))
    assert_size_stride(primals_25, (384, 768), (768, 1))
    assert_size_stride(primals_26, (384, ), (1, ))
    assert_size_stride(primals_27, (384, ), (1, ))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (384, 196), (196, 1))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_31, (196, 192), (192, 1))
    assert_size_stride(primals_32, (196, ), (1, ))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (1536, 384), (384, 1))
    assert_size_stride(primals_36, (1536, ), (1, ))
    assert_size_stride(primals_37, (384, 768), (768, 1))
    assert_size_stride(primals_38, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, 196), (196, 1))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (196, 192), (192, 1))
    assert_size_stride(primals_44, (196, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_47, (1536, 384), (384, 1))
    assert_size_stride(primals_48, (1536, ), (1, ))
    assert_size_stride(primals_49, (384, 768), (768, 1))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_53, (384, 196), (196, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (196, 192), (192, 1))
    assert_size_stride(primals_56, (196, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (1536, 384), (384, 1))
    assert_size_stride(primals_60, (1536, ), (1, ))
    assert_size_stride(primals_61, (384, 768), (768, 1))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_65, (384, 196), (196, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (196, 192), (192, 1))
    assert_size_stride(primals_68, (196, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (1536, 384), (384, 1))
    assert_size_stride(primals_72, (1536, ), (1, ))
    assert_size_stride(primals_73, (384, 768), (768, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (384, 196), (196, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (196, 192), (192, 1))
    assert_size_stride(primals_80, (196, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_82, (384, ), (1, ))
    assert_size_stride(primals_83, (1536, 384), (384, 1))
    assert_size_stride(primals_84, (1536, ), (1, ))
    assert_size_stride(primals_85, (384, 768), (768, 1))
    assert_size_stride(primals_86, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (384, 196), (196, 1))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (196, 192), (192, 1))
    assert_size_stride(primals_92, (196, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (1536, 384), (384, 1))
    assert_size_stride(primals_96, (1536, ), (1, ))
    assert_size_stride(primals_97, (384, 768), (768, 1))
    assert_size_stride(primals_98, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, 196), (196, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (196, 192), (192, 1))
    assert_size_stride(primals_104, (196, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (1536, 384), (384, 1))
    assert_size_stride(primals_108, (1536, ), (1, ))
    assert_size_stride(primals_109, (384, 768), (768, 1))
    assert_size_stride(primals_110, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, 196), (196, 1))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (196, 192), (192, 1))
    assert_size_stride(primals_116, (196, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (1536, 384), (384, 1))
    assert_size_stride(primals_120, (1536, ), (1, ))
    assert_size_stride(primals_121, (384, 768), (768, 1))
    assert_size_stride(primals_122, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (384, 196), (196, 1))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (196, 192), (192, 1))
    assert_size_stride(primals_128, (196, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (1536, 384), (384, 1))
    assert_size_stride(primals_132, (1536, ), (1, ))
    assert_size_stride(primals_133, (384, 768), (768, 1))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (384, 196), (196, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (196, 192), (192, 1))
    assert_size_stride(primals_140, (196, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (1536, 384), (384, 1))
    assert_size_stride(primals_144, (1536, ), (1, ))
    assert_size_stride(primals_145, (384, 768), (768, 1))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, 196), (196, 1))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (196, 192), (192, 1))
    assert_size_stride(primals_152, (196, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, ), (1, ))
    assert_size_stride(primals_155, (1536, 384), (384, 1))
    assert_size_stride(primals_156, (1536, ), (1, ))
    assert_size_stride(primals_157, (384, 768), (768, 1))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, 196), (196, 1))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (196, 192), (192, 1))
    assert_size_stride(primals_164, (196, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (1536, 384), (384, 1))
    assert_size_stride(primals_168, (1536, ), (1, ))
    assert_size_stride(primals_169, (384, 768), (768, 1))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_173, (384, 196), (196, 1))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (196, 192), (192, 1))
    assert_size_stride(primals_176, (196, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (1536, 384), (384, 1))
    assert_size_stride(primals_180, (1536, ), (1, ))
    assert_size_stride(primals_181, (384, 768), (768, 1))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (384, ), (1, ))
    assert_size_stride(primals_185, (384, 196), (196, 1))
    assert_size_stride(primals_186, (384, ), (1, ))
    assert_size_stride(primals_187, (196, 192), (192, 1))
    assert_size_stride(primals_188, (196, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (384, ), (1, ))
    assert_size_stride(primals_191, (1536, 384), (384, 1))
    assert_size_stride(primals_192, (1536, ), (1, ))
    assert_size_stride(primals_193, (384, 768), (768, 1))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_197, (384, 196), (196, 1))
    assert_size_stride(primals_198, (384, ), (1, ))
    assert_size_stride(primals_199, (196, 192), (192, 1))
    assert_size_stride(primals_200, (196, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_203, (1536, 384), (384, 1))
    assert_size_stride(primals_204, (1536, ), (1, ))
    assert_size_stride(primals_205, (384, 768), (768, 1))
    assert_size_stride(primals_206, (384, ), (1, ))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_208, (384, ), (1, ))
    assert_size_stride(primals_209, (384, 196), (196, 1))
    assert_size_stride(primals_210, (384, ), (1, ))
    assert_size_stride(primals_211, (196, 192), (192, 1))
    assert_size_stride(primals_212, (196, ), (1, ))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_214, (384, ), (1, ))
    assert_size_stride(primals_215, (1536, 384), (384, 1))
    assert_size_stride(primals_216, (1536, ), (1, ))
    assert_size_stride(primals_217, (384, 768), (768, 1))
    assert_size_stride(primals_218, (384, ), (1, ))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_220, (384, ), (1, ))
    assert_size_stride(primals_221, (384, 196), (196, 1))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (196, 192), (192, 1))
    assert_size_stride(primals_224, (196, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (384, ), (1, ))
    assert_size_stride(primals_227, (1536, 384), (384, 1))
    assert_size_stride(primals_228, (1536, ), (1, ))
    assert_size_stride(primals_229, (384, 768), (768, 1))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (384, ), (1, ))
    assert_size_stride(primals_233, (384, 196), (196, 1))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (196, 192), (192, 1))
    assert_size_stride(primals_236, (196, ), (1, ))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_238, (384, ), (1, ))
    assert_size_stride(primals_239, (1536, 384), (384, 1))
    assert_size_stride(primals_240, (1536, ), (1, ))
    assert_size_stride(primals_241, (384, 768), (768, 1))
    assert_size_stride(primals_242, (384, ), (1, ))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_245, (384, 196), (196, 1))
    assert_size_stride(primals_246, (384, ), (1, ))
    assert_size_stride(primals_247, (196, 192), (192, 1))
    assert_size_stride(primals_248, (196, ), (1, ))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (384, ), (1, ))
    assert_size_stride(primals_251, (1536, 384), (384, 1))
    assert_size_stride(primals_252, (1536, ), (1, ))
    assert_size_stride(primals_253, (384, 768), (768, 1))
    assert_size_stride(primals_254, (384, ), (1, ))
    assert_size_stride(primals_255, (384, ), (1, ))
    assert_size_stride(primals_256, (384, ), (1, ))
    assert_size_stride(primals_257, (384, 196), (196, 1))
    assert_size_stride(primals_258, (384, ), (1, ))
    assert_size_stride(primals_259, (196, 192), (192, 1))
    assert_size_stride(primals_260, (196, ), (1, ))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_262, (384, ), (1, ))
    assert_size_stride(primals_263, (1536, 384), (384, 1))
    assert_size_stride(primals_264, (1536, ), (1, ))
    assert_size_stride(primals_265, (384, 768), (768, 1))
    assert_size_stride(primals_266, (384, ), (1, ))
    assert_size_stride(primals_267, (384, ), (1, ))
    assert_size_stride(primals_268, (384, ), (1, ))
    assert_size_stride(primals_269, (384, 196), (196, 1))
    assert_size_stride(primals_270, (384, ), (1, ))
    assert_size_stride(primals_271, (196, 192), (192, 1))
    assert_size_stride(primals_272, (196, ), (1, ))
    assert_size_stride(primals_273, (384, ), (1, ))
    assert_size_stride(primals_274, (384, ), (1, ))
    assert_size_stride(primals_275, (1536, 384), (384, 1))
    assert_size_stride(primals_276, (1536, ), (1, ))
    assert_size_stride(primals_277, (384, 768), (768, 1))
    assert_size_stride(primals_278, (384, ), (1, ))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_281, (384, 196), (196, 1))
    assert_size_stride(primals_282, (384, ), (1, ))
    assert_size_stride(primals_283, (196, 192), (192, 1))
    assert_size_stride(primals_284, (196, ), (1, ))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_287, (1536, 384), (384, 1))
    assert_size_stride(primals_288, (1536, ), (1, ))
    assert_size_stride(primals_289, (384, 768), (768, 1))
    assert_size_stride(primals_290, (384, ), (1, ))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_292, (384, ), (1, ))
    assert_size_stride(primals_293, (1000, 384), (384, 1))
    assert_size_stride(primals_294, (1000, ), (1, ))
    assert_size_stride(primals_295, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_295, primals_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf1 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, primals_2, buf1, buf2, buf3, 4704, 128, grid=grid(4704), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf612 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf1, buf2, buf3, buf4, buf5, buf612, 1568, 3, grid=grid(1568), stream=stream0)
        buf7 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_2.run(buf0, primals_2, buf4, buf5, buf7, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf8 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf7, primals_3, primals_4, buf8, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_4
        buf9 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, reinterpret_tensor(primals_5, (196, 384), (1, 196), 0), out=buf9)
        buf10 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___mlp_tokens_act, x_5, x_8], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf9, primals_6, buf10, 589824, grid=grid(589824), stream=stream0)
        buf11 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf10, reinterpret_tensor(primals_7, (192, 196), (1, 192), 0), out=buf11)
        buf12 = buf3; del buf3  # reuse
        buf13 = buf2; del buf2  # reuse
        buf14 = buf1; del buf1  # reuse
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf0, primals_2, buf11, primals_8, buf12, buf13, buf14, 4704, 128, grid=grid(4704), stream=stream0)
        buf15 = buf5; del buf5  # reuse
        buf16 = buf4; del buf4  # reuse
        buf611 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf12, buf13, buf14, buf15, buf16, buf611, 1568, 3, grid=grid(1568), stream=stream0)
        buf18 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_6.run(buf0, primals_2, buf11, primals_8, buf15, buf16, buf18, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf19 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_11], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf18, primals_9, primals_10, buf19, 602112, grid=grid(602112), stream=stream0)
        del primals_10
        buf20 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, buf19, reinterpret_tensor(primals_11, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf20)
        del primals_12
        buf21 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___mlp_channels_act, x_12, x_15], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf20, buf21, 1204224, grid=grid(1204224), stream=stream0)
        buf22 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf21, reinterpret_tensor(primals_13, (768, 384), (1, 768), 0), out=buf22)
        buf23 = reinterpret_tensor(buf0, (8, 196, 384), (75264, 1, 196), 0); del buf0  # reuse
        # Source Nodes: [x_10, x_17], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf23, primals_2, buf11, primals_8, buf22, primals_14, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_14
        del primals_2
        del primals_8
        buf24 = buf14; del buf14  # reuse
        buf25 = buf13; del buf13  # reuse
        buf26 = buf12; del buf12  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf23, buf24, buf25, buf26, 4704, 128, grid=grid(4704), stream=stream0)
        buf27 = buf16; del buf16  # reuse
        buf28 = buf15; del buf15  # reuse
        buf610 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf24, buf25, buf26, buf27, buf28, buf610, 1568, 3, grid=grid(1568), stream=stream0)
        buf30 = reinterpret_tensor(buf22, (8, 196, 384), (75264, 384, 1), 0); del buf22  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf23, buf27, buf28, buf30, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf31 = buf11; del buf11  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf30, primals_15, primals_16, buf31, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_16
        buf32 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, reinterpret_tensor(primals_17, (196, 384), (1, 196), 0), out=buf32)
        buf33 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___mlp_tokens_act, x_19, x_22], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf32, primals_18, buf33, 589824, grid=grid(589824), stream=stream0)
        buf34 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_19, (192, 196), (1, 192), 0), out=buf34)
        buf35 = buf26; del buf26  # reuse
        buf36 = buf25; del buf25  # reuse
        buf37 = buf24; del buf24  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf23, buf34, primals_20, buf35, buf36, buf37, 4704, 128, grid=grid(4704), stream=stream0)
        buf38 = buf28; del buf28  # reuse
        buf39 = buf27; del buf27  # reuse
        buf609 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf35, buf36, buf37, buf38, buf39, buf609, 1568, 3, grid=grid(1568), stream=stream0)
        buf41 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf42 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24, x_25], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf23, buf34, primals_20, buf38, buf39, primals_21, primals_22, buf41, buf42, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_22
        buf43 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, buf42, reinterpret_tensor(primals_23, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf43)
        del primals_24
        buf44 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___mlp_channels_act, x_26, x_29], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf43, buf44, 1204224, grid=grid(1204224), stream=stream0)
        buf45 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_25, (768, 384), (1, 768), 0), out=buf45)
        buf46 = buf23; del buf23  # reuse
        # Source Nodes: [x_24, x_31], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf46, buf34, primals_20, buf45, primals_26, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_20
        del primals_26
        buf47 = buf37; del buf37  # reuse
        buf48 = buf36; del buf36  # reuse
        buf49 = buf35; del buf35  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf46, buf47, buf48, buf49, 4704, 128, grid=grid(4704), stream=stream0)
        buf50 = buf39; del buf39  # reuse
        buf51 = buf38; del buf38  # reuse
        buf608 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf47, buf48, buf49, buf50, buf51, buf608, 1568, 3, grid=grid(1568), stream=stream0)
        buf53 = reinterpret_tensor(buf45, (8, 196, 384), (75264, 384, 1), 0); del buf45  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf46, buf50, buf51, buf53, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf54 = buf34; del buf34  # reuse
        # Source Nodes: [x_32], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf53, primals_27, primals_28, buf54, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_28
        buf55 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.mm]
        extern_kernels.mm(buf54, reinterpret_tensor(primals_29, (196, 384), (1, 196), 0), out=buf55)
        buf56 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___mlp_tokens_act, x_33, x_36], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf55, primals_30, buf56, 589824, grid=grid(589824), stream=stream0)
        buf57 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(primals_31, (192, 196), (1, 192), 0), out=buf57)
        buf58 = buf49; del buf49  # reuse
        buf59 = buf48; del buf48  # reuse
        buf60 = buf47; del buf47  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf46, buf57, primals_32, buf58, buf59, buf60, 4704, 128, grid=grid(4704), stream=stream0)
        buf61 = buf51; del buf51  # reuse
        buf62 = buf50; del buf50  # reuse
        buf607 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf58, buf59, buf60, buf61, buf62, buf607, 1568, 3, grid=grid(1568), stream=stream0)
        buf64 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf46, buf57, primals_32, buf61, buf62, buf64, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf65 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_39], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf64, primals_33, primals_34, buf65, 602112, grid=grid(602112), stream=stream0)
        del primals_34
        buf66 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_36, buf65, reinterpret_tensor(primals_35, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf66)
        del primals_36
        buf67 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___mlp_channels_act, x_40, x_43], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf66, buf67, 1204224, grid=grid(1204224), stream=stream0)
        buf68 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf67, reinterpret_tensor(primals_37, (768, 384), (1, 768), 0), out=buf68)
        buf69 = buf46; del buf46  # reuse
        # Source Nodes: [x_38, x_45], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf69, buf57, primals_32, buf68, primals_38, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_32
        del primals_38
        buf70 = buf60; del buf60  # reuse
        buf71 = buf59; del buf59  # reuse
        buf72 = buf58; del buf58  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf69, buf70, buf71, buf72, 4704, 128, grid=grid(4704), stream=stream0)
        buf73 = buf62; del buf62  # reuse
        buf74 = buf61; del buf61  # reuse
        buf606 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf70, buf71, buf72, buf73, buf74, buf606, 1568, 3, grid=grid(1568), stream=stream0)
        buf76 = reinterpret_tensor(buf68, (8, 196, 384), (75264, 384, 1), 0); del buf68  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf69, buf73, buf74, buf76, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf77 = buf57; del buf57  # reuse
        # Source Nodes: [x_46], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf76, primals_39, primals_40, buf77, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_40
        buf78 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.mm]
        extern_kernels.mm(buf77, reinterpret_tensor(primals_41, (196, 384), (1, 196), 0), out=buf78)
        buf79 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___mlp_tokens_act, x_47, x_50], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf78, primals_42, buf79, 589824, grid=grid(589824), stream=stream0)
        buf80 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf79, reinterpret_tensor(primals_43, (192, 196), (1, 192), 0), out=buf80)
        buf81 = buf72; del buf72  # reuse
        buf82 = buf71; del buf71  # reuse
        buf83 = buf70; del buf70  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf69, buf80, primals_44, buf81, buf82, buf83, 4704, 128, grid=grid(4704), stream=stream0)
        buf84 = buf74; del buf74  # reuse
        buf85 = buf73; del buf73  # reuse
        buf605 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf81, buf82, buf83, buf84, buf85, buf605, 1568, 3, grid=grid(1568), stream=stream0)
        buf87 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf88 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52, x_53], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf69, buf80, primals_44, buf84, buf85, primals_45, primals_46, buf87, buf88, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_46
        buf89 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, buf88, reinterpret_tensor(primals_47, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf89)
        del primals_48
        buf90 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___mlp_channels_act, x_54, x_57], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf89, buf90, 1204224, grid=grid(1204224), stream=stream0)
        buf91 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf90, reinterpret_tensor(primals_49, (768, 384), (1, 768), 0), out=buf91)
        buf92 = buf69; del buf69  # reuse
        # Source Nodes: [x_52, x_59], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf92, buf80, primals_44, buf91, primals_50, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_44
        del primals_50
        buf93 = buf83; del buf83  # reuse
        buf94 = buf82; del buf82  # reuse
        buf95 = buf81; del buf81  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf92, buf93, buf94, buf95, 4704, 128, grid=grid(4704), stream=stream0)
        buf96 = buf85; del buf85  # reuse
        buf97 = buf84; del buf84  # reuse
        buf604 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf93, buf94, buf95, buf96, buf97, buf604, 1568, 3, grid=grid(1568), stream=stream0)
        buf99 = reinterpret_tensor(buf91, (8, 196, 384), (75264, 384, 1), 0); del buf91  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf92, buf96, buf97, buf99, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf100 = buf80; del buf80  # reuse
        # Source Nodes: [x_60], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf99, primals_51, primals_52, buf100, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_52
        buf101 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.mm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_53, (196, 384), (1, 196), 0), out=buf101)
        buf102 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___mlp_tokens_act, x_61, x_64], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf101, primals_54, buf102, 589824, grid=grid(589824), stream=stream0)
        buf103 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf102, reinterpret_tensor(primals_55, (192, 196), (1, 192), 0), out=buf103)
        buf104 = buf95; del buf95  # reuse
        buf105 = buf94; del buf94  # reuse
        buf106 = buf93; del buf93  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf92, buf103, primals_56, buf104, buf105, buf106, 4704, 128, grid=grid(4704), stream=stream0)
        buf107 = buf97; del buf97  # reuse
        buf108 = buf96; del buf96  # reuse
        buf603 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf104, buf105, buf106, buf107, buf108, buf603, 1568, 3, grid=grid(1568), stream=stream0)
        buf110 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf92, buf103, primals_56, buf107, buf108, buf110, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf111 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_67], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf110, primals_57, primals_58, buf111, 602112, grid=grid(602112), stream=stream0)
        del primals_58
        buf112 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, buf111, reinterpret_tensor(primals_59, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf112)
        del primals_60
        buf113 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___mlp_channels_act, x_68, x_71], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf112, buf113, 1204224, grid=grid(1204224), stream=stream0)
        buf114 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf113, reinterpret_tensor(primals_61, (768, 384), (1, 768), 0), out=buf114)
        buf115 = reinterpret_tensor(buf103, (8, 196, 384), (75264, 1, 196), 0); del buf103  # reuse
        # Source Nodes: [x_66, x_73], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf115, buf92, primals_56, buf114, primals_62, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_56
        del primals_62
        buf116 = buf106; del buf106  # reuse
        buf117 = buf105; del buf105  # reuse
        buf118 = buf104; del buf104  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf115, buf116, buf117, buf118, 4704, 128, grid=grid(4704), stream=stream0)
        buf119 = buf108; del buf108  # reuse
        buf120 = buf107; del buf107  # reuse
        buf602 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf116, buf117, buf118, buf119, buf120, buf602, 1568, 3, grid=grid(1568), stream=stream0)
        buf122 = reinterpret_tensor(buf92, (8, 196, 384), (75264, 384, 1), 0); del buf92  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf115, buf119, buf120, buf122, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf123 = reinterpret_tensor(buf114, (3072, 196), (196, 1), 0); del buf114  # reuse
        # Source Nodes: [x_74], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf122, primals_63, primals_64, buf123, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_64
        buf124 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.mm]
        extern_kernels.mm(buf123, reinterpret_tensor(primals_65, (196, 384), (1, 196), 0), out=buf124)
        buf125 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___mlp_tokens_act, x_75, x_78], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf124, primals_66, buf125, 589824, grid=grid(589824), stream=stream0)
        buf126 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf125, reinterpret_tensor(primals_67, (192, 196), (1, 192), 0), out=buf126)
        buf127 = buf118; del buf118  # reuse
        buf128 = buf117; del buf117  # reuse
        buf129 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf115, buf126, primals_68, buf127, buf128, buf129, 4704, 128, grid=grid(4704), stream=stream0)
        buf130 = buf120; del buf120  # reuse
        buf131 = buf119; del buf119  # reuse
        buf601 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf127, buf128, buf129, buf130, buf131, buf601, 1568, 3, grid=grid(1568), stream=stream0)
        buf133 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf134 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80, x_81], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf115, buf126, primals_68, buf130, buf131, primals_69, primals_70, buf133, buf134, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_70
        buf135 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_72, buf134, reinterpret_tensor(primals_71, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf135)
        del primals_72
        buf136 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___mlp_channels_act, x_82, x_85], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf135, buf136, 1204224, grid=grid(1204224), stream=stream0)
        buf137 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf136, reinterpret_tensor(primals_73, (768, 384), (1, 768), 0), out=buf137)
        buf138 = buf115; del buf115  # reuse
        # Source Nodes: [x_80, x_87], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf138, buf126, primals_68, buf137, primals_74, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_68
        del primals_74
        buf139 = buf129; del buf129  # reuse
        buf140 = buf128; del buf128  # reuse
        buf141 = buf127; del buf127  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf138, buf139, buf140, buf141, 4704, 128, grid=grid(4704), stream=stream0)
        buf142 = buf131; del buf131  # reuse
        buf143 = buf130; del buf130  # reuse
        buf600 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf139, buf140, buf141, buf142, buf143, buf600, 1568, 3, grid=grid(1568), stream=stream0)
        buf145 = reinterpret_tensor(buf137, (8, 196, 384), (75264, 384, 1), 0); del buf137  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf138, buf142, buf143, buf145, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf146 = buf126; del buf126  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf145, primals_75, primals_76, buf146, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_76
        buf147 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.mm]
        extern_kernels.mm(buf146, reinterpret_tensor(primals_77, (196, 384), (1, 196), 0), out=buf147)
        buf148 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___mlp_tokens_act, x_89, x_92], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf147, primals_78, buf148, 589824, grid=grid(589824), stream=stream0)
        buf149 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf148, reinterpret_tensor(primals_79, (192, 196), (1, 192), 0), out=buf149)
        buf150 = buf141; del buf141  # reuse
        buf151 = buf140; del buf140  # reuse
        buf152 = buf139; del buf139  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf138, buf149, primals_80, buf150, buf151, buf152, 4704, 128, grid=grid(4704), stream=stream0)
        buf153 = buf143; del buf143  # reuse
        buf154 = buf142; del buf142  # reuse
        buf599 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf150, buf151, buf152, buf153, buf154, buf599, 1568, 3, grid=grid(1568), stream=stream0)
        buf156 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf138, buf149, primals_80, buf153, buf154, buf156, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf157 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_95], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf156, primals_81, primals_82, buf157, 602112, grid=grid(602112), stream=stream0)
        del primals_82
        buf158 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_84, buf157, reinterpret_tensor(primals_83, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf158)
        del primals_84
        buf159 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___mlp_channels_act, x_96, x_99], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf158, buf159, 1204224, grid=grid(1204224), stream=stream0)
        buf160 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf159, reinterpret_tensor(primals_85, (768, 384), (1, 768), 0), out=buf160)
        buf161 = buf138; del buf138  # reuse
        # Source Nodes: [x_101, x_94], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf161, buf149, primals_80, buf160, primals_86, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_80
        del primals_86
        buf162 = buf152; del buf152  # reuse
        buf163 = buf151; del buf151  # reuse
        buf164 = buf150; del buf150  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf161, buf162, buf163, buf164, 4704, 128, grid=grid(4704), stream=stream0)
        buf165 = buf154; del buf154  # reuse
        buf166 = buf153; del buf153  # reuse
        buf598 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf162, buf163, buf164, buf165, buf166, buf598, 1568, 3, grid=grid(1568), stream=stream0)
        buf168 = reinterpret_tensor(buf160, (8, 196, 384), (75264, 384, 1), 0); del buf160  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf161, buf165, buf166, buf168, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf169 = buf149; del buf149  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf168, primals_87, primals_88, buf169, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_88
        buf170 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten.mm]
        extern_kernels.mm(buf169, reinterpret_tensor(primals_89, (196, 384), (1, 196), 0), out=buf170)
        buf171 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___mlp_tokens_act, x_103, x_106], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf170, primals_90, buf171, 589824, grid=grid(589824), stream=stream0)
        buf172 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, reinterpret_tensor(primals_91, (192, 196), (1, 192), 0), out=buf172)
        buf173 = buf164; del buf164  # reuse
        buf174 = buf163; del buf163  # reuse
        buf175 = buf162; del buf162  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf161, buf172, primals_92, buf173, buf174, buf175, 4704, 128, grid=grid(4704), stream=stream0)
        buf176 = buf166; del buf166  # reuse
        buf177 = buf165; del buf165  # reuse
        buf597 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf173, buf174, buf175, buf176, buf177, buf597, 1568, 3, grid=grid(1568), stream=stream0)
        buf179 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf180 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108, x_109], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf161, buf172, primals_92, buf176, buf177, primals_93, primals_94, buf179, buf180, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_94
        buf181 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf180, reinterpret_tensor(primals_95, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf181)
        del primals_96
        buf182 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___mlp_channels_act, x_110, x_113], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf181, buf182, 1204224, grid=grid(1204224), stream=stream0)
        buf183 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf182, reinterpret_tensor(primals_97, (768, 384), (1, 768), 0), out=buf183)
        buf184 = buf161; del buf161  # reuse
        # Source Nodes: [x_108, x_115], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf184, buf172, primals_92, buf183, primals_98, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_92
        del primals_98
        buf185 = buf175; del buf175  # reuse
        buf186 = buf174; del buf174  # reuse
        buf187 = buf173; del buf173  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf184, buf185, buf186, buf187, 4704, 128, grid=grid(4704), stream=stream0)
        buf188 = buf177; del buf177  # reuse
        buf189 = buf176; del buf176  # reuse
        buf596 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf185, buf186, buf187, buf188, buf189, buf596, 1568, 3, grid=grid(1568), stream=stream0)
        buf191 = reinterpret_tensor(buf183, (8, 196, 384), (75264, 384, 1), 0); del buf183  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf184, buf188, buf189, buf191, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf192 = buf172; del buf172  # reuse
        # Source Nodes: [x_116], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf191, primals_99, primals_100, buf192, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_100
        buf193 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.mm]
        extern_kernels.mm(buf192, reinterpret_tensor(primals_101, (196, 384), (1, 196), 0), out=buf193)
        buf194 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___mlp_tokens_act, x_117, x_120], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf193, primals_102, buf194, 589824, grid=grid(589824), stream=stream0)
        buf195 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf194, reinterpret_tensor(primals_103, (192, 196), (1, 192), 0), out=buf195)
        buf196 = buf187; del buf187  # reuse
        buf197 = buf186; del buf186  # reuse
        buf198 = buf185; del buf185  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf184, buf195, primals_104, buf196, buf197, buf198, 4704, 128, grid=grid(4704), stream=stream0)
        buf199 = buf189; del buf189  # reuse
        buf200 = buf188; del buf188  # reuse
        buf595 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf196, buf197, buf198, buf199, buf200, buf595, 1568, 3, grid=grid(1568), stream=stream0)
        buf202 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf184, buf195, primals_104, buf199, buf200, buf202, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf203 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_123], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf202, primals_105, primals_106, buf203, 602112, grid=grid(602112), stream=stream0)
        del primals_106
        buf204 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_108, buf203, reinterpret_tensor(primals_107, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf204)
        del primals_108
        buf205 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___mlp_channels_act, x_124, x_127], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf204, buf205, 1204224, grid=grid(1204224), stream=stream0)
        buf206 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf205, reinterpret_tensor(primals_109, (768, 384), (1, 768), 0), out=buf206)
        buf207 = buf184; del buf184  # reuse
        # Source Nodes: [x_122, x_129], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf207, buf195, primals_104, buf206, primals_110, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_104
        del primals_110
        buf208 = buf198; del buf198  # reuse
        buf209 = buf197; del buf197  # reuse
        buf210 = buf196; del buf196  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf207, buf208, buf209, buf210, 4704, 128, grid=grid(4704), stream=stream0)
        buf211 = buf200; del buf200  # reuse
        buf212 = buf199; del buf199  # reuse
        buf594 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf208, buf209, buf210, buf211, buf212, buf594, 1568, 3, grid=grid(1568), stream=stream0)
        buf214 = reinterpret_tensor(buf206, (8, 196, 384), (75264, 384, 1), 0); del buf206  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf207, buf211, buf212, buf214, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf215 = buf195; del buf195  # reuse
        # Source Nodes: [x_130], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf214, primals_111, primals_112, buf215, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_112
        buf216 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten.mm]
        extern_kernels.mm(buf215, reinterpret_tensor(primals_113, (196, 384), (1, 196), 0), out=buf216)
        buf217 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___mlp_tokens_act, x_131, x_134], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf216, primals_114, buf217, 589824, grid=grid(589824), stream=stream0)
        buf218 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf217, reinterpret_tensor(primals_115, (192, 196), (1, 192), 0), out=buf218)
        buf219 = buf210; del buf210  # reuse
        buf220 = buf209; del buf209  # reuse
        buf221 = buf208; del buf208  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf207, buf218, primals_116, buf219, buf220, buf221, 4704, 128, grid=grid(4704), stream=stream0)
        buf222 = buf212; del buf212  # reuse
        buf223 = buf211; del buf211  # reuse
        buf593 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf219, buf220, buf221, buf222, buf223, buf593, 1568, 3, grid=grid(1568), stream=stream0)
        buf225 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf226 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136, x_137], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf207, buf218, primals_116, buf222, buf223, primals_117, primals_118, buf225, buf226, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_118
        buf227 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf226, reinterpret_tensor(primals_119, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf227)
        del primals_120
        buf228 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___mlp_channels_act, x_138, x_141], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf227, buf228, 1204224, grid=grid(1204224), stream=stream0)
        buf229 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf228, reinterpret_tensor(primals_121, (768, 384), (1, 768), 0), out=buf229)
        buf230 = buf207; del buf207  # reuse
        # Source Nodes: [x_136, x_143], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf230, buf218, primals_116, buf229, primals_122, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_116
        del primals_122
        buf231 = buf221; del buf221  # reuse
        buf232 = buf220; del buf220  # reuse
        buf233 = buf219; del buf219  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf230, buf231, buf232, buf233, 4704, 128, grid=grid(4704), stream=stream0)
        buf234 = buf223; del buf223  # reuse
        buf235 = buf222; del buf222  # reuse
        buf592 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf231, buf232, buf233, buf234, buf235, buf592, 1568, 3, grid=grid(1568), stream=stream0)
        buf237 = reinterpret_tensor(buf229, (8, 196, 384), (75264, 384, 1), 0); del buf229  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf230, buf234, buf235, buf237, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf238 = buf218; del buf218  # reuse
        # Source Nodes: [x_144], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf237, primals_123, primals_124, buf238, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_124
        buf239 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.mm]
        extern_kernels.mm(buf238, reinterpret_tensor(primals_125, (196, 384), (1, 196), 0), out=buf239)
        buf240 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___mlp_tokens_act, x_145, x_148], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf239, primals_126, buf240, 589824, grid=grid(589824), stream=stream0)
        buf241 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf240, reinterpret_tensor(primals_127, (192, 196), (1, 192), 0), out=buf241)
        buf242 = buf233; del buf233  # reuse
        buf243 = buf232; del buf232  # reuse
        buf244 = buf231; del buf231  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf230, buf241, primals_128, buf242, buf243, buf244, 4704, 128, grid=grid(4704), stream=stream0)
        buf245 = buf235; del buf235  # reuse
        buf246 = buf234; del buf234  # reuse
        buf591 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf242, buf243, buf244, buf245, buf246, buf591, 1568, 3, grid=grid(1568), stream=stream0)
        buf248 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf230, buf241, primals_128, buf245, buf246, buf248, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf249 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_151], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf248, primals_129, primals_130, buf249, 602112, grid=grid(602112), stream=stream0)
        del primals_130
        buf250 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf249, reinterpret_tensor(primals_131, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf250)
        del primals_132
        buf251 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___mlp_channels_act, x_152, x_155], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf250, buf251, 1204224, grid=grid(1204224), stream=stream0)
        buf252 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf251, reinterpret_tensor(primals_133, (768, 384), (1, 768), 0), out=buf252)
        buf253 = buf230; del buf230  # reuse
        # Source Nodes: [x_150, x_157], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf253, buf241, primals_128, buf252, primals_134, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_128
        del primals_134
        buf254 = buf244; del buf244  # reuse
        buf255 = buf243; del buf243  # reuse
        buf256 = buf242; del buf242  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf253, buf254, buf255, buf256, 4704, 128, grid=grid(4704), stream=stream0)
        buf257 = buf246; del buf246  # reuse
        buf258 = buf245; del buf245  # reuse
        buf590 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf254, buf255, buf256, buf257, buf258, buf590, 1568, 3, grid=grid(1568), stream=stream0)
        buf260 = reinterpret_tensor(buf252, (8, 196, 384), (75264, 384, 1), 0); del buf252  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf253, buf257, buf258, buf260, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf261 = buf241; del buf241  # reuse
        # Source Nodes: [x_158], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf260, primals_135, primals_136, buf261, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_136
        buf262 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.mm]
        extern_kernels.mm(buf261, reinterpret_tensor(primals_137, (196, 384), (1, 196), 0), out=buf262)
        buf263 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___mlp_tokens_act, x_159, x_162], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf262, primals_138, buf263, 589824, grid=grid(589824), stream=stream0)
        buf264 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf263, reinterpret_tensor(primals_139, (192, 196), (1, 192), 0), out=buf264)
        buf265 = buf256; del buf256  # reuse
        buf266 = buf255; del buf255  # reuse
        buf267 = buf254; del buf254  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf253, buf264, primals_140, buf265, buf266, buf267, 4704, 128, grid=grid(4704), stream=stream0)
        buf268 = buf258; del buf258  # reuse
        buf269 = buf257; del buf257  # reuse
        buf589 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf265, buf266, buf267, buf268, buf269, buf589, 1568, 3, grid=grid(1568), stream=stream0)
        buf271 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf272 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164, x_165], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf253, buf264, primals_140, buf268, buf269, primals_141, primals_142, buf271, buf272, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_142
        buf273 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf272, reinterpret_tensor(primals_143, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf273)
        del primals_144
        buf274 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___mlp_channels_act, x_166, x_169], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf273, buf274, 1204224, grid=grid(1204224), stream=stream0)
        buf275 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf274, reinterpret_tensor(primals_145, (768, 384), (1, 768), 0), out=buf275)
        buf276 = buf253; del buf253  # reuse
        # Source Nodes: [x_164, x_171], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf276, buf264, primals_140, buf275, primals_146, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_140
        del primals_146
        buf277 = buf267; del buf267  # reuse
        buf278 = buf266; del buf266  # reuse
        buf279 = buf265; del buf265  # reuse
        # Source Nodes: [getattr_l__mod___blocks___12___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf276, buf277, buf278, buf279, 4704, 128, grid=grid(4704), stream=stream0)
        buf280 = buf269; del buf269  # reuse
        buf281 = buf268; del buf268  # reuse
        buf588 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___12___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf277, buf278, buf279, buf280, buf281, buf588, 1568, 3, grid=grid(1568), stream=stream0)
        buf283 = reinterpret_tensor(buf275, (8, 196, 384), (75264, 384, 1), 0); del buf275  # reuse
        # Source Nodes: [getattr_l__mod___blocks___12___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf276, buf280, buf281, buf283, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf284 = buf264; del buf264  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf283, primals_147, primals_148, buf284, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_148
        buf285 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.mm]
        extern_kernels.mm(buf284, reinterpret_tensor(primals_149, (196, 384), (1, 196), 0), out=buf285)
        buf286 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___12___mlp_tokens_act, x_173, x_176], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf285, primals_150, buf286, 589824, grid=grid(589824), stream=stream0)
        buf287 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf286, reinterpret_tensor(primals_151, (192, 196), (1, 192), 0), out=buf287)
        buf288 = buf279; del buf279  # reuse
        buf289 = buf278; del buf278  # reuse
        buf290 = buf277; del buf277  # reuse
        # Source Nodes: [getattr_l__mod___blocks___12___norm2, x_178], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf276, buf287, primals_152, buf288, buf289, buf290, 4704, 128, grid=grid(4704), stream=stream0)
        buf291 = buf281; del buf281  # reuse
        buf292 = buf280; del buf280  # reuse
        buf587 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___12___norm2, x_178], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf288, buf289, buf290, buf291, buf292, buf587, 1568, 3, grid=grid(1568), stream=stream0)
        buf294 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___12___norm2, x_178], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf276, buf287, primals_152, buf291, buf292, buf294, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf295 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___12___norm2, x_179], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf294, primals_153, primals_154, buf295, 602112, grid=grid(602112), stream=stream0)
        del primals_154
        buf296 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_156, buf295, reinterpret_tensor(primals_155, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf296)
        del primals_156
        buf297 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___12___mlp_channels_act, x_180, x_183], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf296, buf297, 1204224, grid=grid(1204224), stream=stream0)
        buf298 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_157, (768, 384), (1, 768), 0), out=buf298)
        buf299 = buf276; del buf276  # reuse
        # Source Nodes: [x_178, x_185], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf299, buf287, primals_152, buf298, primals_158, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_152
        del primals_158
        buf300 = buf290; del buf290  # reuse
        buf301 = buf289; del buf289  # reuse
        buf302 = buf288; del buf288  # reuse
        # Source Nodes: [getattr_l__mod___blocks___13___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf299, buf300, buf301, buf302, 4704, 128, grid=grid(4704), stream=stream0)
        buf303 = buf292; del buf292  # reuse
        buf304 = buf291; del buf291  # reuse
        buf586 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___13___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf300, buf301, buf302, buf303, buf304, buf586, 1568, 3, grid=grid(1568), stream=stream0)
        buf306 = reinterpret_tensor(buf298, (8, 196, 384), (75264, 384, 1), 0); del buf298  # reuse
        # Source Nodes: [getattr_l__mod___blocks___13___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf299, buf303, buf304, buf306, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf307 = buf287; del buf287  # reuse
        # Source Nodes: [x_186], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf306, primals_159, primals_160, buf307, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_160
        buf308 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten.mm]
        extern_kernels.mm(buf307, reinterpret_tensor(primals_161, (196, 384), (1, 196), 0), out=buf308)
        buf309 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___13___mlp_tokens_act, x_187, x_190], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf308, primals_162, buf309, 589824, grid=grid(589824), stream=stream0)
        buf310 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf309, reinterpret_tensor(primals_163, (192, 196), (1, 192), 0), out=buf310)
        buf311 = buf302; del buf302  # reuse
        buf312 = buf301; del buf301  # reuse
        buf313 = buf300; del buf300  # reuse
        # Source Nodes: [getattr_l__mod___blocks___13___norm2, x_192], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf299, buf310, primals_164, buf311, buf312, buf313, 4704, 128, grid=grid(4704), stream=stream0)
        buf314 = buf304; del buf304  # reuse
        buf315 = buf303; del buf303  # reuse
        buf585 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___13___norm2, x_192], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf311, buf312, buf313, buf314, buf315, buf585, 1568, 3, grid=grid(1568), stream=stream0)
        buf317 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf318 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___13___norm2, x_192, x_193], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf299, buf310, primals_164, buf314, buf315, primals_165, primals_166, buf317, buf318, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_166
        buf319 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_168, buf318, reinterpret_tensor(primals_167, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf319)
        del primals_168
        buf320 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___13___mlp_channels_act, x_194, x_197], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf319, buf320, 1204224, grid=grid(1204224), stream=stream0)
        buf321 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf320, reinterpret_tensor(primals_169, (768, 384), (1, 768), 0), out=buf321)
        buf322 = buf299; del buf299  # reuse
        # Source Nodes: [x_192, x_199], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf322, buf310, primals_164, buf321, primals_170, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_164
        del primals_170
        buf323 = buf313; del buf313  # reuse
        buf324 = buf312; del buf312  # reuse
        buf325 = buf311; del buf311  # reuse
        # Source Nodes: [getattr_l__mod___blocks___14___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf322, buf323, buf324, buf325, 4704, 128, grid=grid(4704), stream=stream0)
        buf326 = buf315; del buf315  # reuse
        buf327 = buf314; del buf314  # reuse
        buf584 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___14___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf323, buf324, buf325, buf326, buf327, buf584, 1568, 3, grid=grid(1568), stream=stream0)
        buf329 = reinterpret_tensor(buf321, (8, 196, 384), (75264, 384, 1), 0); del buf321  # reuse
        # Source Nodes: [getattr_l__mod___blocks___14___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf322, buf326, buf327, buf329, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf330 = buf310; del buf310  # reuse
        # Source Nodes: [x_200], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf329, primals_171, primals_172, buf330, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_172
        buf331 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten.mm]
        extern_kernels.mm(buf330, reinterpret_tensor(primals_173, (196, 384), (1, 196), 0), out=buf331)
        buf332 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___14___mlp_tokens_act, x_201, x_204], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf331, primals_174, buf332, 589824, grid=grid(589824), stream=stream0)
        buf333 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf332, reinterpret_tensor(primals_175, (192, 196), (1, 192), 0), out=buf333)
        buf334 = buf325; del buf325  # reuse
        buf335 = buf324; del buf324  # reuse
        buf336 = buf323; del buf323  # reuse
        # Source Nodes: [getattr_l__mod___blocks___14___norm2, x_206], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf322, buf333, primals_176, buf334, buf335, buf336, 4704, 128, grid=grid(4704), stream=stream0)
        buf337 = buf327; del buf327  # reuse
        buf338 = buf326; del buf326  # reuse
        buf583 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___14___norm2, x_206], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf334, buf335, buf336, buf337, buf338, buf583, 1568, 3, grid=grid(1568), stream=stream0)
        buf340 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___14___norm2, x_206], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf322, buf333, primals_176, buf337, buf338, buf340, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf341 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___14___norm2, x_207], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf340, primals_177, primals_178, buf341, 602112, grid=grid(602112), stream=stream0)
        del primals_178
        buf342 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_180, buf341, reinterpret_tensor(primals_179, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf342)
        del primals_180
        buf343 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___14___mlp_channels_act, x_208, x_211], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf342, buf343, 1204224, grid=grid(1204224), stream=stream0)
        buf344 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf343, reinterpret_tensor(primals_181, (768, 384), (1, 768), 0), out=buf344)
        buf345 = buf322; del buf322  # reuse
        # Source Nodes: [x_206, x_213], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf345, buf333, primals_176, buf344, primals_182, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_176
        del primals_182
        buf346 = buf336; del buf336  # reuse
        buf347 = buf335; del buf335  # reuse
        buf348 = buf334; del buf334  # reuse
        # Source Nodes: [getattr_l__mod___blocks___15___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf345, buf346, buf347, buf348, 4704, 128, grid=grid(4704), stream=stream0)
        buf349 = buf338; del buf338  # reuse
        buf350 = buf337; del buf337  # reuse
        buf582 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___15___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf346, buf347, buf348, buf349, buf350, buf582, 1568, 3, grid=grid(1568), stream=stream0)
        buf352 = reinterpret_tensor(buf344, (8, 196, 384), (75264, 384, 1), 0); del buf344  # reuse
        # Source Nodes: [getattr_l__mod___blocks___15___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf345, buf349, buf350, buf352, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf353 = buf333; del buf333  # reuse
        # Source Nodes: [x_214], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf352, primals_183, primals_184, buf353, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_184
        buf354 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214], Original ATen: [aten.mm]
        extern_kernels.mm(buf353, reinterpret_tensor(primals_185, (196, 384), (1, 196), 0), out=buf354)
        buf355 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___15___mlp_tokens_act, x_215, x_218], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf354, primals_186, buf355, 589824, grid=grid(589824), stream=stream0)
        buf356 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf355, reinterpret_tensor(primals_187, (192, 196), (1, 192), 0), out=buf356)
        buf357 = buf348; del buf348  # reuse
        buf358 = buf347; del buf347  # reuse
        buf359 = buf346; del buf346  # reuse
        # Source Nodes: [getattr_l__mod___blocks___15___norm2, x_220], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf345, buf356, primals_188, buf357, buf358, buf359, 4704, 128, grid=grid(4704), stream=stream0)
        buf360 = buf350; del buf350  # reuse
        buf361 = buf349; del buf349  # reuse
        buf581 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___15___norm2, x_220], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf357, buf358, buf359, buf360, buf361, buf581, 1568, 3, grid=grid(1568), stream=stream0)
        buf363 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf364 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___15___norm2, x_220, x_221], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf345, buf356, primals_188, buf360, buf361, primals_189, primals_190, buf363, buf364, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_190
        buf365 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_192, buf364, reinterpret_tensor(primals_191, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf365)
        del primals_192
        buf366 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___15___mlp_channels_act, x_222, x_225], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf365, buf366, 1204224, grid=grid(1204224), stream=stream0)
        buf367 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_193, (768, 384), (1, 768), 0), out=buf367)
        buf368 = buf345; del buf345  # reuse
        # Source Nodes: [x_220, x_227], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf368, buf356, primals_188, buf367, primals_194, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_188
        del primals_194
        buf369 = buf359; del buf359  # reuse
        buf370 = buf358; del buf358  # reuse
        buf371 = buf357; del buf357  # reuse
        # Source Nodes: [getattr_l__mod___blocks___16___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf368, buf369, buf370, buf371, 4704, 128, grid=grid(4704), stream=stream0)
        buf372 = buf361; del buf361  # reuse
        buf373 = buf360; del buf360  # reuse
        buf580 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___16___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf369, buf370, buf371, buf372, buf373, buf580, 1568, 3, grid=grid(1568), stream=stream0)
        buf375 = reinterpret_tensor(buf367, (8, 196, 384), (75264, 384, 1), 0); del buf367  # reuse
        # Source Nodes: [getattr_l__mod___blocks___16___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf368, buf372, buf373, buf375, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf376 = buf356; del buf356  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf375, primals_195, primals_196, buf376, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_196
        buf377 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten.mm]
        extern_kernels.mm(buf376, reinterpret_tensor(primals_197, (196, 384), (1, 196), 0), out=buf377)
        buf378 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___16___mlp_tokens_act, x_229, x_232], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf377, primals_198, buf378, 589824, grid=grid(589824), stream=stream0)
        buf379 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf378, reinterpret_tensor(primals_199, (192, 196), (1, 192), 0), out=buf379)
        buf380 = buf371; del buf371  # reuse
        buf381 = buf370; del buf370  # reuse
        buf382 = buf369; del buf369  # reuse
        # Source Nodes: [getattr_l__mod___blocks___16___norm2, x_234], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf368, buf379, primals_200, buf380, buf381, buf382, 4704, 128, grid=grid(4704), stream=stream0)
        buf383 = buf373; del buf373  # reuse
        buf384 = buf372; del buf372  # reuse
        buf579 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___16___norm2, x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf380, buf381, buf382, buf383, buf384, buf579, 1568, 3, grid=grid(1568), stream=stream0)
        buf386 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___16___norm2, x_234], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf368, buf379, primals_200, buf383, buf384, buf386, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf387 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___16___norm2, x_235], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf386, primals_201, primals_202, buf387, 602112, grid=grid(602112), stream=stream0)
        del primals_202
        buf388 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_204, buf387, reinterpret_tensor(primals_203, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf388)
        del primals_204
        buf389 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___16___mlp_channels_act, x_236, x_239], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf388, buf389, 1204224, grid=grid(1204224), stream=stream0)
        buf390 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf389, reinterpret_tensor(primals_205, (768, 384), (1, 768), 0), out=buf390)
        buf391 = buf368; del buf368  # reuse
        # Source Nodes: [x_234, x_241], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf391, buf379, primals_200, buf390, primals_206, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_200
        del primals_206
        buf392 = buf382; del buf382  # reuse
        buf393 = buf381; del buf381  # reuse
        buf394 = buf380; del buf380  # reuse
        # Source Nodes: [getattr_l__mod___blocks___17___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf391, buf392, buf393, buf394, 4704, 128, grid=grid(4704), stream=stream0)
        buf395 = buf384; del buf384  # reuse
        buf396 = buf383; del buf383  # reuse
        buf578 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___17___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf392, buf393, buf394, buf395, buf396, buf578, 1568, 3, grid=grid(1568), stream=stream0)
        buf398 = reinterpret_tensor(buf390, (8, 196, 384), (75264, 384, 1), 0); del buf390  # reuse
        # Source Nodes: [getattr_l__mod___blocks___17___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf391, buf395, buf396, buf398, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf399 = buf379; del buf379  # reuse
        # Source Nodes: [x_242], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf398, primals_207, primals_208, buf399, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_208
        buf400 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.mm]
        extern_kernels.mm(buf399, reinterpret_tensor(primals_209, (196, 384), (1, 196), 0), out=buf400)
        buf401 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___17___mlp_tokens_act, x_243, x_246], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf400, primals_210, buf401, 589824, grid=grid(589824), stream=stream0)
        buf402 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf401, reinterpret_tensor(primals_211, (192, 196), (1, 192), 0), out=buf402)
        buf403 = buf394; del buf394  # reuse
        buf404 = buf393; del buf393  # reuse
        buf405 = buf392; del buf392  # reuse
        # Source Nodes: [getattr_l__mod___blocks___17___norm2, x_248], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf391, buf402, primals_212, buf403, buf404, buf405, 4704, 128, grid=grid(4704), stream=stream0)
        buf406 = buf396; del buf396  # reuse
        buf407 = buf395; del buf395  # reuse
        buf577 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___17___norm2, x_248], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf403, buf404, buf405, buf406, buf407, buf577, 1568, 3, grid=grid(1568), stream=stream0)
        buf409 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf410 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___17___norm2, x_248, x_249], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf391, buf402, primals_212, buf406, buf407, primals_213, primals_214, buf409, buf410, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_214
        buf411 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_216, buf410, reinterpret_tensor(primals_215, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf411)
        del primals_216
        buf412 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___17___mlp_channels_act, x_250, x_253], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf411, buf412, 1204224, grid=grid(1204224), stream=stream0)
        buf413 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf412, reinterpret_tensor(primals_217, (768, 384), (1, 768), 0), out=buf413)
        buf414 = buf391; del buf391  # reuse
        # Source Nodes: [x_248, x_255], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf414, buf402, primals_212, buf413, primals_218, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_212
        del primals_218
        buf415 = buf405; del buf405  # reuse
        buf416 = buf404; del buf404  # reuse
        buf417 = buf403; del buf403  # reuse
        # Source Nodes: [getattr_l__mod___blocks___18___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf414, buf415, buf416, buf417, 4704, 128, grid=grid(4704), stream=stream0)
        buf418 = buf407; del buf407  # reuse
        buf419 = buf406; del buf406  # reuse
        buf576 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___18___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf415, buf416, buf417, buf418, buf419, buf576, 1568, 3, grid=grid(1568), stream=stream0)
        buf421 = reinterpret_tensor(buf413, (8, 196, 384), (75264, 384, 1), 0); del buf413  # reuse
        # Source Nodes: [getattr_l__mod___blocks___18___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf414, buf418, buf419, buf421, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf422 = buf402; del buf402  # reuse
        # Source Nodes: [x_256], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf421, primals_219, primals_220, buf422, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_220
        buf423 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten.mm]
        extern_kernels.mm(buf422, reinterpret_tensor(primals_221, (196, 384), (1, 196), 0), out=buf423)
        buf424 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___18___mlp_tokens_act, x_257, x_260], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf423, primals_222, buf424, 589824, grid=grid(589824), stream=stream0)
        buf425 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf424, reinterpret_tensor(primals_223, (192, 196), (1, 192), 0), out=buf425)
        buf426 = buf417; del buf417  # reuse
        buf427 = buf416; del buf416  # reuse
        buf428 = buf415; del buf415  # reuse
        # Source Nodes: [getattr_l__mod___blocks___18___norm2, x_262], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf414, buf425, primals_224, buf426, buf427, buf428, 4704, 128, grid=grid(4704), stream=stream0)
        buf429 = buf419; del buf419  # reuse
        buf430 = buf418; del buf418  # reuse
        buf575 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___18___norm2, x_262], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf426, buf427, buf428, buf429, buf430, buf575, 1568, 3, grid=grid(1568), stream=stream0)
        buf432 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___18___norm2, x_262], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf414, buf425, primals_224, buf429, buf430, buf432, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf433 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___18___norm2, x_263], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf432, primals_225, primals_226, buf433, 602112, grid=grid(602112), stream=stream0)
        del primals_226
        buf434 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_228, buf433, reinterpret_tensor(primals_227, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf434)
        del primals_228
        buf435 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___18___mlp_channels_act, x_264, x_267], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf434, buf435, 1204224, grid=grid(1204224), stream=stream0)
        buf436 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf435, reinterpret_tensor(primals_229, (768, 384), (1, 768), 0), out=buf436)
        buf437 = buf414; del buf414  # reuse
        # Source Nodes: [x_262, x_269], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf437, buf425, primals_224, buf436, primals_230, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_224
        del primals_230
        buf438 = buf428; del buf428  # reuse
        buf439 = buf427; del buf427  # reuse
        buf440 = buf426; del buf426  # reuse
        # Source Nodes: [getattr_l__mod___blocks___19___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf437, buf438, buf439, buf440, 4704, 128, grid=grid(4704), stream=stream0)
        buf441 = buf430; del buf430  # reuse
        buf442 = buf429; del buf429  # reuse
        buf574 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___19___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf438, buf439, buf440, buf441, buf442, buf574, 1568, 3, grid=grid(1568), stream=stream0)
        buf444 = reinterpret_tensor(buf436, (8, 196, 384), (75264, 384, 1), 0); del buf436  # reuse
        # Source Nodes: [getattr_l__mod___blocks___19___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf437, buf441, buf442, buf444, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf445 = buf425; del buf425  # reuse
        # Source Nodes: [x_270], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf444, primals_231, primals_232, buf445, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_232
        buf446 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten.mm]
        extern_kernels.mm(buf445, reinterpret_tensor(primals_233, (196, 384), (1, 196), 0), out=buf446)
        buf447 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___19___mlp_tokens_act, x_271, x_274], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf446, primals_234, buf447, 589824, grid=grid(589824), stream=stream0)
        buf448 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf447, reinterpret_tensor(primals_235, (192, 196), (1, 192), 0), out=buf448)
        buf449 = buf440; del buf440  # reuse
        buf450 = buf439; del buf439  # reuse
        buf451 = buf438; del buf438  # reuse
        # Source Nodes: [getattr_l__mod___blocks___19___norm2, x_276], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf437, buf448, primals_236, buf449, buf450, buf451, 4704, 128, grid=grid(4704), stream=stream0)
        buf452 = buf442; del buf442  # reuse
        buf453 = buf441; del buf441  # reuse
        buf573 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___19___norm2, x_276], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf449, buf450, buf451, buf452, buf453, buf573, 1568, 3, grid=grid(1568), stream=stream0)
        buf455 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf456 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___19___norm2, x_276, x_277], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf437, buf448, primals_236, buf452, buf453, primals_237, primals_238, buf455, buf456, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_238
        buf457 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_277], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_240, buf456, reinterpret_tensor(primals_239, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf457)
        del primals_240
        buf458 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___19___mlp_channels_act, x_278, x_281], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf457, buf458, 1204224, grid=grid(1204224), stream=stream0)
        buf459 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf458, reinterpret_tensor(primals_241, (768, 384), (1, 768), 0), out=buf459)
        buf460 = buf437; del buf437  # reuse
        # Source Nodes: [x_276, x_283], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf460, buf448, primals_236, buf459, primals_242, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_236
        del primals_242
        buf461 = buf451; del buf451  # reuse
        buf462 = buf450; del buf450  # reuse
        buf463 = buf449; del buf449  # reuse
        # Source Nodes: [getattr_l__mod___blocks___20___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf460, buf461, buf462, buf463, 4704, 128, grid=grid(4704), stream=stream0)
        buf464 = buf453; del buf453  # reuse
        buf465 = buf452; del buf452  # reuse
        buf572 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___20___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf461, buf462, buf463, buf464, buf465, buf572, 1568, 3, grid=grid(1568), stream=stream0)
        buf467 = reinterpret_tensor(buf459, (8, 196, 384), (75264, 384, 1), 0); del buf459  # reuse
        # Source Nodes: [getattr_l__mod___blocks___20___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf460, buf464, buf465, buf467, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf468 = buf448; del buf448  # reuse
        # Source Nodes: [x_284], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf467, primals_243, primals_244, buf468, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_244
        buf469 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.mm]
        extern_kernels.mm(buf468, reinterpret_tensor(primals_245, (196, 384), (1, 196), 0), out=buf469)
        buf470 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___20___mlp_tokens_act, x_285, x_288], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf469, primals_246, buf470, 589824, grid=grid(589824), stream=stream0)
        buf471 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf470, reinterpret_tensor(primals_247, (192, 196), (1, 192), 0), out=buf471)
        buf472 = buf463; del buf463  # reuse
        buf473 = buf462; del buf462  # reuse
        buf474 = buf461; del buf461  # reuse
        # Source Nodes: [getattr_l__mod___blocks___20___norm2, x_290], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf460, buf471, primals_248, buf472, buf473, buf474, 4704, 128, grid=grid(4704), stream=stream0)
        buf475 = buf465; del buf465  # reuse
        buf476 = buf464; del buf464  # reuse
        buf571 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___20___norm2, x_290], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf472, buf473, buf474, buf475, buf476, buf571, 1568, 3, grid=grid(1568), stream=stream0)
        buf478 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___20___norm2, x_290], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf460, buf471, primals_248, buf475, buf476, buf478, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf479 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___20___norm2, x_291], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf478, primals_249, primals_250, buf479, 602112, grid=grid(602112), stream=stream0)
        del primals_250
        buf480 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_252, buf479, reinterpret_tensor(primals_251, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf480)
        del primals_252
        buf481 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___20___mlp_channels_act, x_292, x_295], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf480, buf481, 1204224, grid=grid(1204224), stream=stream0)
        buf482 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf481, reinterpret_tensor(primals_253, (768, 384), (1, 768), 0), out=buf482)
        buf483 = buf460; del buf460  # reuse
        # Source Nodes: [x_290, x_297], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf483, buf471, primals_248, buf482, primals_254, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_248
        del primals_254
        buf484 = buf474; del buf474  # reuse
        buf485 = buf473; del buf473  # reuse
        buf486 = buf472; del buf472  # reuse
        # Source Nodes: [getattr_l__mod___blocks___21___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf483, buf484, buf485, buf486, 4704, 128, grid=grid(4704), stream=stream0)
        buf487 = buf476; del buf476  # reuse
        buf488 = buf475; del buf475  # reuse
        buf570 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___21___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf484, buf485, buf486, buf487, buf488, buf570, 1568, 3, grid=grid(1568), stream=stream0)
        buf490 = reinterpret_tensor(buf482, (8, 196, 384), (75264, 384, 1), 0); del buf482  # reuse
        # Source Nodes: [getattr_l__mod___blocks___21___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf483, buf487, buf488, buf490, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf491 = buf471; del buf471  # reuse
        # Source Nodes: [x_298], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf490, primals_255, primals_256, buf491, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_256
        buf492 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten.mm]
        extern_kernels.mm(buf491, reinterpret_tensor(primals_257, (196, 384), (1, 196), 0), out=buf492)
        buf493 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___21___mlp_tokens_act, x_299, x_302], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf492, primals_258, buf493, 589824, grid=grid(589824), stream=stream0)
        buf494 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf493, reinterpret_tensor(primals_259, (192, 196), (1, 192), 0), out=buf494)
        buf495 = buf486; del buf486  # reuse
        buf496 = buf485; del buf485  # reuse
        buf497 = buf484; del buf484  # reuse
        # Source Nodes: [getattr_l__mod___blocks___21___norm2, x_304], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf483, buf494, primals_260, buf495, buf496, buf497, 4704, 128, grid=grid(4704), stream=stream0)
        buf498 = buf488; del buf488  # reuse
        buf499 = buf487; del buf487  # reuse
        buf569 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___21___norm2, x_304], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf495, buf496, buf497, buf498, buf499, buf569, 1568, 3, grid=grid(1568), stream=stream0)
        buf501 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf502 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___21___norm2, x_304, x_305], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf483, buf494, primals_260, buf498, buf499, primals_261, primals_262, buf501, buf502, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_262
        buf503 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_305], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_264, buf502, reinterpret_tensor(primals_263, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf503)
        del primals_264
        buf504 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___21___mlp_channels_act, x_306, x_309], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf503, buf504, 1204224, grid=grid(1204224), stream=stream0)
        buf505 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf504, reinterpret_tensor(primals_265, (768, 384), (1, 768), 0), out=buf505)
        buf506 = buf483; del buf483  # reuse
        # Source Nodes: [x_304, x_311], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf506, buf494, primals_260, buf505, primals_266, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_260
        del primals_266
        buf507 = buf497; del buf497  # reuse
        buf508 = buf496; del buf496  # reuse
        buf509 = buf495; del buf495  # reuse
        # Source Nodes: [getattr_l__mod___blocks___22___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf506, buf507, buf508, buf509, 4704, 128, grid=grid(4704), stream=stream0)
        buf510 = buf499; del buf499  # reuse
        buf511 = buf498; del buf498  # reuse
        buf568 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___22___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf507, buf508, buf509, buf510, buf511, buf568, 1568, 3, grid=grid(1568), stream=stream0)
        buf513 = reinterpret_tensor(buf505, (8, 196, 384), (75264, 384, 1), 0); del buf505  # reuse
        # Source Nodes: [getattr_l__mod___blocks___22___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf506, buf510, buf511, buf513, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf514 = buf494; del buf494  # reuse
        # Source Nodes: [x_312], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf513, primals_267, primals_268, buf514, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_268
        buf515 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten.mm]
        extern_kernels.mm(buf514, reinterpret_tensor(primals_269, (196, 384), (1, 196), 0), out=buf515)
        buf516 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___22___mlp_tokens_act, x_313, x_316], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf515, primals_270, buf516, 589824, grid=grid(589824), stream=stream0)
        buf517 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf516, reinterpret_tensor(primals_271, (192, 196), (1, 192), 0), out=buf517)
        buf518 = buf509; del buf509  # reuse
        buf519 = buf508; del buf508  # reuse
        buf520 = buf507; del buf507  # reuse
        # Source Nodes: [getattr_l__mod___blocks___22___norm2, x_318], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf506, buf517, primals_272, buf518, buf519, buf520, 4704, 128, grid=grid(4704), stream=stream0)
        buf521 = buf511; del buf511  # reuse
        buf522 = buf510; del buf510  # reuse
        buf567 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___22___norm2, x_318], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf518, buf519, buf520, buf521, buf522, buf567, 1568, 3, grid=grid(1568), stream=stream0)
        buf524 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___22___norm2, x_318], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_16.run(buf506, buf517, primals_272, buf521, buf522, buf524, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf525 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___22___norm2, x_319], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_7.run(buf524, primals_273, primals_274, buf525, 602112, grid=grid(602112), stream=stream0)
        del primals_274
        buf526 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_319], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_276, buf525, reinterpret_tensor(primals_275, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf526)
        del primals_276
        buf527 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___22___mlp_channels_act, x_320, x_323], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf526, buf527, 1204224, grid=grid(1204224), stream=stream0)
        buf528 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf527, reinterpret_tensor(primals_277, (768, 384), (1, 768), 0), out=buf528)
        buf529 = buf506; del buf506  # reuse
        # Source Nodes: [x_318, x_325], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf529, buf517, primals_272, buf528, primals_278, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_272
        del primals_278
        buf530 = buf520; del buf520  # reuse
        buf531 = buf519; del buf519  # reuse
        buf532 = buf518; del buf518  # reuse
        # Source Nodes: [getattr_l__mod___blocks___23___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_10.run(buf529, buf530, buf531, buf532, 4704, 128, grid=grid(4704), stream=stream0)
        buf533 = buf522; del buf522  # reuse
        buf534 = buf521; del buf521  # reuse
        buf566 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___23___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf530, buf531, buf532, buf533, buf534, buf566, 1568, 3, grid=grid(1568), stream=stream0)
        buf536 = reinterpret_tensor(buf528, (8, 196, 384), (75264, 384, 1), 0); del buf528  # reuse
        # Source Nodes: [getattr_l__mod___blocks___23___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf529, buf533, buf534, buf536, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf537 = buf517; del buf517  # reuse
        # Source Nodes: [x_326], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_3.run(buf536, primals_279, primals_280, buf537, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_280
        buf538 = empty((3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten.mm]
        extern_kernels.mm(buf537, reinterpret_tensor(primals_281, (196, 384), (1, 196), 0), out=buf538)
        buf539 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___23___mlp_tokens_act, x_327, x_330], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_4.run(buf538, primals_282, buf539, 589824, grid=grid(589824), stream=stream0)
        buf540 = empty((3072, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf539, reinterpret_tensor(primals_283, (192, 196), (1, 192), 0), out=buf540)
        buf541 = buf532; del buf532  # reuse
        buf542 = buf531; del buf531  # reuse
        buf543 = buf530; del buf530  # reuse
        # Source Nodes: [getattr_l__mod___blocks___23___norm2, x_332], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf529, buf540, primals_284, buf541, buf542, buf543, 4704, 128, grid=grid(4704), stream=stream0)
        buf544 = buf534; del buf534  # reuse
        buf545 = buf533; del buf533  # reuse
        buf565 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___23___norm2, x_332], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf541, buf542, buf543, buf544, buf545, buf565, 1568, 3, grid=grid(1568), stream=stream0)
        buf547 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        buf548 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___23___norm2, x_332, x_333], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_13.run(buf529, buf540, primals_284, buf544, buf545, primals_285, primals_286, buf547, buf548, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_286
        buf549 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_333], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_288, buf548, reinterpret_tensor(primals_287, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf549)
        del primals_288
        buf550 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___23___mlp_channels_act, x_334, x_337], Original ATen: [aten.mul, aten.silu, aten.view]
        triton_poi_fused_mul_silu_view_8.run(buf549, buf550, 1204224, grid=grid(1204224), stream=stream0)
        buf551 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf550, reinterpret_tensor(primals_289, (768, 384), (1, 768), 0), out=buf551)
        buf552 = buf529; del buf529  # reuse
        # Source Nodes: [x_332, x_340, x_342], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_14.run(buf552, buf540, primals_284, buf551, primals_290, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del buf540
        del primals_284
        del primals_290
        buf553 = buf543; del buf543  # reuse
        buf554 = buf542; del buf542  # reuse
        buf555 = buf541; del buf541  # reuse
        # Source Nodes: [x_342], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_15.run(buf552, buf553, buf554, buf555, 4704, 128, grid=grid(4704), stream=stream0)
        buf556 = buf545; del buf545  # reuse
        buf557 = buf544; del buf544  # reuse
        buf564 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_342], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf553, buf554, buf555, buf556, buf557, buf564, 1568, 3, grid=grid(1568), stream=stream0)
        del buf553
        del buf554
        del buf555
        buf559 = reinterpret_tensor(buf551, (8, 196, 384), (75264, 384, 1), 0); del buf551  # reuse
        # Source Nodes: [x_342], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_11.run(buf552, buf556, buf557, buf559, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del buf552
        del buf556
        del buf557
        buf560 = empty_strided((8, 384, 2), (768, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_342, x_343], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_red_fused_mean_native_layer_norm_19.run(buf559, primals_291, primals_292, buf560, 6144, 98, grid=grid(6144), stream=stream0)
        del primals_292
        buf561 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf562 = buf561; del buf561  # reuse
        # Source Nodes: [x_342, x_343], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_20.run(buf562, buf560, 3072, 2, grid=grid(3072), stream=stream0)
        del buf560
        buf563 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_294, buf562, reinterpret_tensor(primals_293, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf563)
        del primals_294
        buf613 = reinterpret_tensor(buf9, (8, 384, 384), (147456, 384, 1), 0); del buf9  # reuse
        # Source Nodes: [x_4], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf613, primals_6, 1179648, grid=grid(1179648), stream=stream0)
        del primals_6
        buf614 = reinterpret_tensor(buf32, (8, 384, 384), (147456, 384, 1), 0); del buf32  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf614, primals_18, 1179648, grid=grid(1179648), stream=stream0)
        del primals_18
        buf615 = reinterpret_tensor(buf55, (8, 384, 384), (147456, 384, 1), 0); del buf55  # reuse
        # Source Nodes: [x_32], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf615, primals_30, 1179648, grid=grid(1179648), stream=stream0)
        del primals_30
        buf616 = reinterpret_tensor(buf78, (8, 384, 384), (147456, 384, 1), 0); del buf78  # reuse
        # Source Nodes: [x_46], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf616, primals_42, 1179648, grid=grid(1179648), stream=stream0)
        del primals_42
        buf617 = reinterpret_tensor(buf101, (8, 384, 384), (147456, 384, 1), 0); del buf101  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf617, primals_54, 1179648, grid=grid(1179648), stream=stream0)
        del primals_54
        buf618 = reinterpret_tensor(buf124, (8, 384, 384), (147456, 384, 1), 0); del buf124  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf618, primals_66, 1179648, grid=grid(1179648), stream=stream0)
        del primals_66
        buf619 = reinterpret_tensor(buf147, (8, 384, 384), (147456, 384, 1), 0); del buf147  # reuse
        # Source Nodes: [x_88], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf619, primals_78, 1179648, grid=grid(1179648), stream=stream0)
        del primals_78
        buf620 = reinterpret_tensor(buf170, (8, 384, 384), (147456, 384, 1), 0); del buf170  # reuse
        # Source Nodes: [x_102], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf620, primals_90, 1179648, grid=grid(1179648), stream=stream0)
        del primals_90
        buf621 = reinterpret_tensor(buf193, (8, 384, 384), (147456, 384, 1), 0); del buf193  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf621, primals_102, 1179648, grid=grid(1179648), stream=stream0)
        del primals_102
        buf622 = reinterpret_tensor(buf216, (8, 384, 384), (147456, 384, 1), 0); del buf216  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf622, primals_114, 1179648, grid=grid(1179648), stream=stream0)
        del primals_114
        buf623 = reinterpret_tensor(buf239, (8, 384, 384), (147456, 384, 1), 0); del buf239  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf623, primals_126, 1179648, grid=grid(1179648), stream=stream0)
        del primals_126
        buf624 = reinterpret_tensor(buf262, (8, 384, 384), (147456, 384, 1), 0); del buf262  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf624, primals_138, 1179648, grid=grid(1179648), stream=stream0)
        del primals_138
        buf625 = reinterpret_tensor(buf285, (8, 384, 384), (147456, 384, 1), 0); del buf285  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf625, primals_150, 1179648, grid=grid(1179648), stream=stream0)
        del primals_150
        buf626 = reinterpret_tensor(buf308, (8, 384, 384), (147456, 384, 1), 0); del buf308  # reuse
        # Source Nodes: [x_186], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf626, primals_162, 1179648, grid=grid(1179648), stream=stream0)
        del primals_162
        buf627 = reinterpret_tensor(buf331, (8, 384, 384), (147456, 384, 1), 0); del buf331  # reuse
        # Source Nodes: [x_200], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf627, primals_174, 1179648, grid=grid(1179648), stream=stream0)
        del primals_174
        buf628 = reinterpret_tensor(buf354, (8, 384, 384), (147456, 384, 1), 0); del buf354  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf628, primals_186, 1179648, grid=grid(1179648), stream=stream0)
        del primals_186
        buf629 = reinterpret_tensor(buf377, (8, 384, 384), (147456, 384, 1), 0); del buf377  # reuse
        # Source Nodes: [x_228], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf629, primals_198, 1179648, grid=grid(1179648), stream=stream0)
        del primals_198
        buf630 = reinterpret_tensor(buf400, (8, 384, 384), (147456, 384, 1), 0); del buf400  # reuse
        # Source Nodes: [x_242], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf630, primals_210, 1179648, grid=grid(1179648), stream=stream0)
        del primals_210
        buf631 = reinterpret_tensor(buf423, (8, 384, 384), (147456, 384, 1), 0); del buf423  # reuse
        # Source Nodes: [x_256], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf631, primals_222, 1179648, grid=grid(1179648), stream=stream0)
        del primals_222
        buf632 = reinterpret_tensor(buf446, (8, 384, 384), (147456, 384, 1), 0); del buf446  # reuse
        # Source Nodes: [x_270], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf632, primals_234, 1179648, grid=grid(1179648), stream=stream0)
        del primals_234
        buf633 = reinterpret_tensor(buf469, (8, 384, 384), (147456, 384, 1), 0); del buf469  # reuse
        # Source Nodes: [x_284], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf633, primals_246, 1179648, grid=grid(1179648), stream=stream0)
        del primals_246
        buf634 = reinterpret_tensor(buf492, (8, 384, 384), (147456, 384, 1), 0); del buf492  # reuse
        # Source Nodes: [x_298], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf634, primals_258, 1179648, grid=grid(1179648), stream=stream0)
        del primals_258
        buf635 = reinterpret_tensor(buf515, (8, 384, 384), (147456, 384, 1), 0); del buf515  # reuse
        # Source Nodes: [x_312], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf635, primals_270, 1179648, grid=grid(1179648), stream=stream0)
        del primals_270
        buf636 = reinterpret_tensor(buf538, (8, 384, 384), (147456, 384, 1), 0); del buf538  # reuse
        # Source Nodes: [x_326], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf636, primals_282, 1179648, grid=grid(1179648), stream=stream0)
        del primals_282
        return (buf563, primals_1, primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_159, primals_165, primals_171, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_231, primals_237, primals_243, primals_249, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, primals_295, buf7, buf8, reinterpret_tensor(buf613, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf613, (8, 384, 192), (147456, 384, 1), 192), buf10, buf18, buf19, reinterpret_tensor(buf20, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf20, (8, 196, 768), (301056, 1536, 1), 768), buf21, buf30, buf31, reinterpret_tensor(buf614, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf614, (8, 384, 192), (147456, 384, 1), 192), buf33, buf41, buf42, reinterpret_tensor(buf43, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf43, (8, 196, 768), (301056, 1536, 1), 768), buf44, buf53, buf54, reinterpret_tensor(buf615, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf615, (8, 384, 192), (147456, 384, 1), 192), buf56, buf64, buf65, reinterpret_tensor(buf66, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf66, (8, 196, 768), (301056, 1536, 1), 768), buf67, buf76, buf77, reinterpret_tensor(buf616, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf616, (8, 384, 192), (147456, 384, 1), 192), buf79, buf87, buf88, reinterpret_tensor(buf89, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf89, (8, 196, 768), (301056, 1536, 1), 768), buf90, buf99, buf100, reinterpret_tensor(buf617, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf617, (8, 384, 192), (147456, 384, 1), 192), buf102, buf110, buf111, reinterpret_tensor(buf112, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf112, (8, 196, 768), (301056, 1536, 1), 768), buf113, buf122, buf123, reinterpret_tensor(buf618, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf618, (8, 384, 192), (147456, 384, 1), 192), buf125, buf133, buf134, reinterpret_tensor(buf135, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf135, (8, 196, 768), (301056, 1536, 1), 768), buf136, buf145, buf146, reinterpret_tensor(buf619, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf619, (8, 384, 192), (147456, 384, 1), 192), buf148, buf156, buf157, reinterpret_tensor(buf158, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf158, (8, 196, 768), (301056, 1536, 1), 768), buf159, buf168, buf169, reinterpret_tensor(buf620, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf620, (8, 384, 192), (147456, 384, 1), 192), buf171, buf179, buf180, reinterpret_tensor(buf181, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf181, (8, 196, 768), (301056, 1536, 1), 768), buf182, buf191, buf192, reinterpret_tensor(buf621, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf621, (8, 384, 192), (147456, 384, 1), 192), buf194, buf202, buf203, reinterpret_tensor(buf204, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf204, (8, 196, 768), (301056, 1536, 1), 768), buf205, buf214, buf215, reinterpret_tensor(buf622, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf622, (8, 384, 192), (147456, 384, 1), 192), buf217, buf225, buf226, reinterpret_tensor(buf227, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf227, (8, 196, 768), (301056, 1536, 1), 768), buf228, buf237, buf238, reinterpret_tensor(buf623, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf623, (8, 384, 192), (147456, 384, 1), 192), buf240, buf248, buf249, reinterpret_tensor(buf250, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf250, (8, 196, 768), (301056, 1536, 1), 768), buf251, buf260, buf261, reinterpret_tensor(buf624, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf624, (8, 384, 192), (147456, 384, 1), 192), buf263, buf271, buf272, reinterpret_tensor(buf273, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf273, (8, 196, 768), (301056, 1536, 1), 768), buf274, buf283, buf284, reinterpret_tensor(buf625, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf625, (8, 384, 192), (147456, 384, 1), 192), buf286, buf294, buf295, reinterpret_tensor(buf296, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf296, (8, 196, 768), (301056, 1536, 1), 768), buf297, buf306, buf307, reinterpret_tensor(buf626, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf626, (8, 384, 192), (147456, 384, 1), 192), buf309, buf317, buf318, reinterpret_tensor(buf319, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf319, (8, 196, 768), (301056, 1536, 1), 768), buf320, buf329, buf330, reinterpret_tensor(buf627, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf627, (8, 384, 192), (147456, 384, 1), 192), buf332, buf340, buf341, reinterpret_tensor(buf342, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf342, (8, 196, 768), (301056, 1536, 1), 768), buf343, buf352, buf353, reinterpret_tensor(buf628, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf628, (8, 384, 192), (147456, 384, 1), 192), buf355, buf363, buf364, reinterpret_tensor(buf365, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf365, (8, 196, 768), (301056, 1536, 1), 768), buf366, buf375, buf376, reinterpret_tensor(buf629, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf629, (8, 384, 192), (147456, 384, 1), 192), buf378, buf386, buf387, reinterpret_tensor(buf388, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf388, (8, 196, 768), (301056, 1536, 1), 768), buf389, buf398, buf399, reinterpret_tensor(buf630, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf630, (8, 384, 192), (147456, 384, 1), 192), buf401, buf409, buf410, reinterpret_tensor(buf411, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf411, (8, 196, 768), (301056, 1536, 1), 768), buf412, buf421, buf422, reinterpret_tensor(buf631, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf631, (8, 384, 192), (147456, 384, 1), 192), buf424, buf432, buf433, reinterpret_tensor(buf434, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf434, (8, 196, 768), (301056, 1536, 1), 768), buf435, buf444, buf445, reinterpret_tensor(buf632, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf632, (8, 384, 192), (147456, 384, 1), 192), buf447, buf455, buf456, reinterpret_tensor(buf457, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf457, (8, 196, 768), (301056, 1536, 1), 768), buf458, buf467, buf468, reinterpret_tensor(buf633, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf633, (8, 384, 192), (147456, 384, 1), 192), buf470, buf478, buf479, reinterpret_tensor(buf480, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf480, (8, 196, 768), (301056, 1536, 1), 768), buf481, buf490, buf491, reinterpret_tensor(buf634, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf634, (8, 384, 192), (147456, 384, 1), 192), buf493, buf501, buf502, reinterpret_tensor(buf503, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf503, (8, 196, 768), (301056, 1536, 1), 768), buf504, buf513, buf514, reinterpret_tensor(buf635, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf635, (8, 384, 192), (147456, 384, 1), 192), buf516, buf524, buf525, reinterpret_tensor(buf526, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf526, (8, 196, 768), (301056, 1536, 1), 768), buf527, buf536, buf537, reinterpret_tensor(buf636, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf636, (8, 384, 192), (147456, 384, 1), 192), buf539, buf547, buf548, reinterpret_tensor(buf549, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf549, (8, 196, 768), (301056, 1536, 1), 768), buf550, buf559, buf562, reinterpret_tensor(primals_293, (1000, 384), (384, 1), 0), buf564, reinterpret_tensor(primals_289, (384, 768), (768, 1), 0), reinterpret_tensor(primals_287, (1536, 384), (384, 1), 0), buf565, reinterpret_tensor(primals_283, (196, 192), (192, 1), 0), reinterpret_tensor(primals_281, (384, 196), (196, 1), 0), buf566, reinterpret_tensor(primals_277, (384, 768), (768, 1), 0), reinterpret_tensor(primals_275, (1536, 384), (384, 1), 0), buf567, reinterpret_tensor(primals_271, (196, 192), (192, 1), 0), reinterpret_tensor(primals_269, (384, 196), (196, 1), 0), buf568, reinterpret_tensor(primals_265, (384, 768), (768, 1), 0), reinterpret_tensor(primals_263, (1536, 384), (384, 1), 0), buf569, reinterpret_tensor(primals_259, (196, 192), (192, 1), 0), reinterpret_tensor(primals_257, (384, 196), (196, 1), 0), buf570, reinterpret_tensor(primals_253, (384, 768), (768, 1), 0), reinterpret_tensor(primals_251, (1536, 384), (384, 1), 0), buf571, reinterpret_tensor(primals_247, (196, 192), (192, 1), 0), reinterpret_tensor(primals_245, (384, 196), (196, 1), 0), buf572, reinterpret_tensor(primals_241, (384, 768), (768, 1), 0), reinterpret_tensor(primals_239, (1536, 384), (384, 1), 0), buf573, reinterpret_tensor(primals_235, (196, 192), (192, 1), 0), reinterpret_tensor(primals_233, (384, 196), (196, 1), 0), buf574, reinterpret_tensor(primals_229, (384, 768), (768, 1), 0), reinterpret_tensor(primals_227, (1536, 384), (384, 1), 0), buf575, reinterpret_tensor(primals_223, (196, 192), (192, 1), 0), reinterpret_tensor(primals_221, (384, 196), (196, 1), 0), buf576, reinterpret_tensor(primals_217, (384, 768), (768, 1), 0), reinterpret_tensor(primals_215, (1536, 384), (384, 1), 0), buf577, reinterpret_tensor(primals_211, (196, 192), (192, 1), 0), reinterpret_tensor(primals_209, (384, 196), (196, 1), 0), buf578, reinterpret_tensor(primals_205, (384, 768), (768, 1), 0), reinterpret_tensor(primals_203, (1536, 384), (384, 1), 0), buf579, reinterpret_tensor(primals_199, (196, 192), (192, 1), 0), reinterpret_tensor(primals_197, (384, 196), (196, 1), 0), buf580, reinterpret_tensor(primals_193, (384, 768), (768, 1), 0), reinterpret_tensor(primals_191, (1536, 384), (384, 1), 0), buf581, reinterpret_tensor(primals_187, (196, 192), (192, 1), 0), reinterpret_tensor(primals_185, (384, 196), (196, 1), 0), buf582, reinterpret_tensor(primals_181, (384, 768), (768, 1), 0), reinterpret_tensor(primals_179, (1536, 384), (384, 1), 0), buf583, reinterpret_tensor(primals_175, (196, 192), (192, 1), 0), reinterpret_tensor(primals_173, (384, 196), (196, 1), 0), buf584, reinterpret_tensor(primals_169, (384, 768), (768, 1), 0), reinterpret_tensor(primals_167, (1536, 384), (384, 1), 0), buf585, reinterpret_tensor(primals_163, (196, 192), (192, 1), 0), reinterpret_tensor(primals_161, (384, 196), (196, 1), 0), buf586, reinterpret_tensor(primals_157, (384, 768), (768, 1), 0), reinterpret_tensor(primals_155, (1536, 384), (384, 1), 0), buf587, reinterpret_tensor(primals_151, (196, 192), (192, 1), 0), reinterpret_tensor(primals_149, (384, 196), (196, 1), 0), buf588, reinterpret_tensor(primals_145, (384, 768), (768, 1), 0), reinterpret_tensor(primals_143, (1536, 384), (384, 1), 0), buf589, reinterpret_tensor(primals_139, (196, 192), (192, 1), 0), reinterpret_tensor(primals_137, (384, 196), (196, 1), 0), buf590, reinterpret_tensor(primals_133, (384, 768), (768, 1), 0), reinterpret_tensor(primals_131, (1536, 384), (384, 1), 0), buf591, reinterpret_tensor(primals_127, (196, 192), (192, 1), 0), reinterpret_tensor(primals_125, (384, 196), (196, 1), 0), buf592, reinterpret_tensor(primals_121, (384, 768), (768, 1), 0), reinterpret_tensor(primals_119, (1536, 384), (384, 1), 0), buf593, reinterpret_tensor(primals_115, (196, 192), (192, 1), 0), reinterpret_tensor(primals_113, (384, 196), (196, 1), 0), buf594, reinterpret_tensor(primals_109, (384, 768), (768, 1), 0), reinterpret_tensor(primals_107, (1536, 384), (384, 1), 0), buf595, reinterpret_tensor(primals_103, (196, 192), (192, 1), 0), reinterpret_tensor(primals_101, (384, 196), (196, 1), 0), buf596, reinterpret_tensor(primals_97, (384, 768), (768, 1), 0), reinterpret_tensor(primals_95, (1536, 384), (384, 1), 0), buf597, reinterpret_tensor(primals_91, (196, 192), (192, 1), 0), reinterpret_tensor(primals_89, (384, 196), (196, 1), 0), buf598, reinterpret_tensor(primals_85, (384, 768), (768, 1), 0), reinterpret_tensor(primals_83, (1536, 384), (384, 1), 0), buf599, reinterpret_tensor(primals_79, (196, 192), (192, 1), 0), reinterpret_tensor(primals_77, (384, 196), (196, 1), 0), buf600, reinterpret_tensor(primals_73, (384, 768), (768, 1), 0), reinterpret_tensor(primals_71, (1536, 384), (384, 1), 0), buf601, reinterpret_tensor(primals_67, (196, 192), (192, 1), 0), reinterpret_tensor(primals_65, (384, 196), (196, 1), 0), buf602, reinterpret_tensor(primals_61, (384, 768), (768, 1), 0), reinterpret_tensor(primals_59, (1536, 384), (384, 1), 0), buf603, reinterpret_tensor(primals_55, (196, 192), (192, 1), 0), reinterpret_tensor(primals_53, (384, 196), (196, 1), 0), buf604, reinterpret_tensor(primals_49, (384, 768), (768, 1), 0), reinterpret_tensor(primals_47, (1536, 384), (384, 1), 0), buf605, reinterpret_tensor(primals_43, (196, 192), (192, 1), 0), reinterpret_tensor(primals_41, (384, 196), (196, 1), 0), buf606, reinterpret_tensor(primals_37, (384, 768), (768, 1), 0), reinterpret_tensor(primals_35, (1536, 384), (384, 1), 0), buf607, reinterpret_tensor(primals_31, (196, 192), (192, 1), 0), reinterpret_tensor(primals_29, (384, 196), (196, 1), 0), buf608, reinterpret_tensor(primals_25, (384, 768), (768, 1), 0), reinterpret_tensor(primals_23, (1536, 384), (384, 1), 0), buf609, reinterpret_tensor(primals_19, (196, 192), (192, 1), 0), reinterpret_tensor(primals_17, (384, 196), (196, 1), 0), buf610, reinterpret_tensor(primals_13, (384, 768), (768, 1), 0), reinterpret_tensor(primals_11, (1536, 384), (384, 1), 0), buf611, reinterpret_tensor(primals_7, (196, 192), (192, 1), 0), reinterpret_tensor(primals_5, (384, 196), (196, 1), 0), buf612, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmixer_24_224', benchmark_compiled_module)
