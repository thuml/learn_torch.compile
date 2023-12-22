
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


# kernel path: /tmp/torchinductor_youkaichao/f4/cf4wwwd5pkemlaiejqqericd32mbjwv5zm6jodfbgcoxflt476h5.py
# Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm]
# l__mod___norm1_proj => var_mean
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((4*(x1 % 14)) + (56*((((4*((r3 + (128*x0)) // 96)) + (((r3 + (128*x0)) // 24) % 4)) // 4) % 4)) + (56*(tl.where(((4*(x1 // 14)) + ((((4*((r3 + (128*x0)) // 96)) + (((r3 + (128*x0)) // 24) % 4)) // 4) % 4)) >= 0, 0, 56))) + (224*(x1 // 14)) + (3136*((((4*((r3 + (128*x0)) // 96)) + (16*((r3 + (128*x0)) % 24)) + (((r3 + (128*x0)) // 24) % 4)) // 16) % 24)) + (75264*x2) + (((r3 + (128*x0)) // 24) % 4) + (tl.where(((4*(x1 % 14)) + (((r3 + (128*x0)) // 24) % 4)) >= 0, 0, 56))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((((4*((r3 + (128*x0)) // 96)) + (16*((r3 + (128*x0)) % 24)) + (((r3 + (128*x0)) // 24) % 4)) // 16) % 24), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((16*((r3 + (128*x0)) % 24)) + ((r3 + (128*x0)) // 24)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x5), tmp6, xmask)
    tl.store(out_ptr1 + (x5), tmp7, xmask)
    tl.store(out_ptr2 + (x5), tmp8, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/bh/cbh2tk5as3oigoj7muyvbcyog7hzsu4eqs2rb6isooho2gzh7k6v.py
# Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm]
# l__mod___norm1_proj => var_mean
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czv5d7aw2mducuolzz4dxlh5mo7uiio7jl6kw4oyibarjbi2pyvc.py
# Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm]
# l__mod___norm1_proj => add_3, add_4, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384) % 196
    x2 = (xindex // 75264)
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((4*(x1 % 14)) + (56*((((4*(x0 // 96)) + ((x0 // 24) % 4)) // 4) % 4)) + (56*(tl.where(((4*(x1 // 14)) + ((((4*(x0 // 96)) + ((x0 // 24) % 4)) // 4) % 4)) >= 0, 0, 56))) + (224*(x1 // 14)) + (3136*((((4*(x0 // 96)) + (16*(x0 % 24)) + ((x0 // 24) % 4)) // 16) % 24)) + (75264*x2) + ((x0 // 24) % 4) + (tl.where(((4*(x1 % 14)) + ((x0 // 24) % 4)) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((4*(x0 // 96)) + (16*(x0 % 24)) + ((x0 // 24) % 4)) // 16) % 24), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*(x0 % 24)) + (x0 // 24)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepwf75akgli7yztoqavxhh5r7qfrckc6xl6fh27is43cngyacze.py
# Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm]
# patch_embed => var_mean_1
triton_per_fused_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cglci2hbghyiytytlyjrtocuqux6cowtrie3homjkxgc7335w4zc.py
# Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_norm_in => clone_4, var_mean_2
triton_per_fused_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((4*((x1 % 196) % 14)) + (56*((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) + (56*(tl.where(((4*((x1 % 196) // 14)) + ((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) >= 0, 0, 56))) + (224*((x1 % 196) // 14)) + (3136*((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24)) + (75264*(x1 // 196)) + (x0 % 4) + (tl.where(((4*((x1 % 196) % 14)) + (x0 % 4)) >= 0, 0, 56))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (16*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdwixgpvolcfklxn2grqrpzalslghssbmhjjpdlsowf2nt6wf4r.py
# Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_0_norm_in => add_8, add_9, clone_4, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
triton_poi_fused_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37632
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + ((4*((y1 % 196) % 14)) + (56*((((4*(x2 // 4)) + (x2 % 4)) // 4) % 4)) + (56*(tl.where(((4*((y1 % 196) // 14)) + ((((4*(x2 // 4)) + (x2 % 4)) // 4) % 4)) >= 0, 0, 56))) + (224*((y1 % 196) // 14)) + (3136*((((4*(x2 // 4)) + (16*y0) + (x2 % 4)) // 16) % 24)) + (75264*(y1 // 196)) + (x2 % 4) + (tl.where(((4*((y1 % 196) % 14)) + (x2 % 4)) >= 0, 0, 56))), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((4*(x2 // 4)) + (16*y0) + (x2 % 4)) // 16) % 24), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (16*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (16*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 24.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (y0 + (24*x2) + (384*y1)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3szexnileoxisq5rkq27ys6zeqky4zljkpdb5xmjpvr27o7mcc.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_5
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = (xindex // 6) % 16
    x2 = (xindex // 96) % 4
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*x2) + (48*x1) + (768*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcy5dw5gjs3wwwovhbn32vaeus5lrspcplmnjy6kjbpj2v6zzqa.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_6
triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37632
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (24 + y0 + (48*x2) + (768*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7f3loxwdmrpiyb27lkipi6463pmboqj7drprto6pbgfua7d2bcl.py
# Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
# attn => mul_6
# attn_1 => amax, div, exp, sub_3, sum_1
triton_per_fused__softmax_mul_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask, other=0.0)
    tmp1 = 0.408248290463863
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (16*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/clyu4yabuf7vfl5apzniwdf23t7ssp3zdisfr4fwdaumgtieewer.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_7
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = (xindex // 6) % 16
    x2 = (xindex // 96) % 4
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*x2) + (24*x1) + (384*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbyhjp7lp2iwmlexcvplhcyfb6ky437o4gsbbglb7ebnzmvfiol.py
# Source Nodes: [x_5], Original ATen: [aten.clone]
# x_5 => clone_8
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = (xindex // 6) % 4
    x2 = (xindex // 24) % 16
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*x2) + (96*x1) + (384*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rqf4by34bgphmzih4impy2rstbetl6c2qtjaq52vku22pys4gv.py
# Source Nodes: [l__mod___blocks_0_norm_mlp_in, pixel_embed_1], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_0_norm_mlp_in => add_11, add_12, clone_9, mul_7, mul_8, rsqrt_3, sub_4, var_mean_3
# pixel_embed_1 => add_10
triton_per_fused_add_native_layer_norm_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((4*((x1 % 196) % 14)) + (56*((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) + (56*(tl.where(((4*((x1 % 196) // 14)) + ((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) >= 0, 0, 56))) + (224*((x1 % 196) // 14)) + (3136*((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24)) + (75264*(x1 // 196)) + (x0 % 4) + (tl.where(((4*((x1 % 196) % 14)) + (x0 % 4)) >= 0, 0, 56))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (16*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (24*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 24.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (24*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (24*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3djhrkpkzdiwmeohdyovzmgnm756q5lijzea3nqowtgkw7fwui.py
# Source Nodes: [x_9], Original ATen: [aten.gelu]
# x_9 => add_13, erf, mul_10, mul_11, mul_9
triton_poi_fused_gelu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajxnpkzlcdz3x6es3ndtmqho2qaghmaqgjc2wwin55nnurheqbp.py
# Source Nodes: [l__mod___blocks_0_norm1_proj, l__mod___blocks_1_norm_in, pixel_embed_3], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_0_norm1_proj => add_15, add_16, clone_12, mul_12, mul_13, rsqrt_4, sub_5, var_mean_4
# l__mod___blocks_1_norm_in => add_25, add_26, clone_19, mul_22, mul_23, rsqrt_7, sub_9, var_mean_7
# pixel_embed_3 => add_14
triton_per_fused_add_native_layer_norm_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 24.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(out_ptr4 + (r1 + (24*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (24*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedl4fbuzemz6ynjhnynwbvrvurrrnpwxrjvbb6a2rqktzeawugt.py
# Source Nodes: [cat_25, patch_embed_2], Original ATen: [aten.add, aten.cat]
# cat_25 => cat
# patch_embed_2 => add_7
triton_poi_fused_add_cat_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 384) % 197
    x0 = xindex % 384
    x2 = (xindex // 75648)
    x3 = xindex % 75648
    x4 = xindex
    tmp28 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-384) + x3 + (75264*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-1) + x1 + (196*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-1) + x1 + (196*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 384.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cesomqqf6m5mwrteh563itvmtwmctnxbxifgl5or57ll2t55e5p6.py
# Source Nodes: [cat_24, l__mod___blocks_0_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_24 => cat_1
# l__mod___blocks_0_norm_out => add_18, add_19, mul_14, mul_15, rsqrt_5, sub_6, var_mean_5
triton_per_fused_cat_native_layer_norm_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp42 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (75648*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-384) + r2 + (384*x0) + (75264*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 384, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 384.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6t3qqsu2lqkxrpycjkwfzau4drgosyuvczfbyw44vkjvomx742.py
# Source Nodes: [matmul_2], Original ATen: [aten.clone]
# matmul_2 => clone_13
triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 6
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (151296*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccywouvg6xxvpeyxggaqsrw4c4sirc6xoodba72dmwqii2cthzqv.py
# Source Nodes: [matmul_2], Original ATen: [aten.clone]
# matmul_2 => clone_14
triton_poi_fused_clone_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (384 + y0 + (768*x2) + (151296*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdn4t5tahf3b2vwkk5nbzqmv5gcelibswfmpcc7uf2itmpdz5wdc.py
# Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
# attn_3 => mul_16
# attn_4 => amax_1, div_1, exp_1, sub_7, sum_2
triton_per_fused__softmax_mul_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9456
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (197*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/is/cismbwbligpsebamqsgjeckdyl27xo5rh7ouw27reh47wayg2zup.py
# Source Nodes: [matmul_3], Original ATen: [aten.clone]
# matmul_3 => clone_15
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 6
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (384*x1) + (75648*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehnqfq6x3tjktottnnbvu6kratm25casl2v5euhata2lntyxbzc.py
# Source Nodes: [x_14], Original ATen: [aten.clone]
# x_14 => clone_16
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x2 = (xindex // 384) % 197
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (12608*x1) + (75648*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2fjq6halhw2ny3i3ijd4uaixfzlzauwjvz5nyeo2tqsvnygh6p.py
# Source Nodes: [cat_24, l__mod___blocks_0_norm_mlp, patch_embed_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_24 => cat_1
# l__mod___blocks_0_norm_mlp => add_21, add_22, mul_17, mul_18, rsqrt_6, sub_8, var_mean_6
# patch_embed_5 => add_20
triton_per_fused_add_cat_native_layer_norm_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp19 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (75648*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + ((-384) + r2 + (384*x0) + (75264*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 384, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 384.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgdkq7pqotx53ikbfrwte4ajjq5pm5nsrnrvvismqp4k65gmxzn.py
# Source Nodes: [l__mod___blocks_1_norm_mlp_in, pixel_embed_3, pixel_embed_4], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_1_norm_mlp_in => add_28, add_29, clone_24, mul_25, mul_26, rsqrt_8, sub_11, var_mean_8
# pixel_embed_3 => add_14
# pixel_embed_4 => add_27
triton_per_fused_add_native_layer_norm_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_22', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 24.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfica2y63j2ozifqzgehkn5tji56rfcqrnqq25sthmj5euq5sp2.py
# Source Nodes: [x_18], Original ATen: [aten.gelu]
# x_18 => add_23, erf_1, mul_19, mul_20, mul_21
triton_poi_fused_gelu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2420736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6fncjirfjpr6thxl3cofnufolgte7tkcwqhhuzxxdc32be6z5w.py
# Source Nodes: [cat_23, l__mod___blocks_1_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_23 => cat_2
# l__mod___blocks_1_norm_out => add_35, add_36, mul_32, mul_33, rsqrt_10, sub_13, var_mean_10
triton_per_fused_cat_native_layer_norm_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 197)
    tmp50 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 197, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & tmp12 & xmask, other=0.0)
    tmp16 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & tmp12 & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.load(in_ptr2 + ((-384) + r2 + (384*x0) + (75264*x1)), rmask & tmp12 & xmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp12, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp11, tmp25)
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tl.full([1], 384, tl.int32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 / tmp35
    tmp37 = tmp27 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp43 = tmp26 - tmp36
    tmp44 = 384.0
    tmp45 = tmp42 / tmp44
    tmp46 = 1e-05
    tmp47 = tmp45 + tmp46
    tmp48 = tl.math.rsqrt(tmp47)
    tmp49 = tmp43 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp26, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp53, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci27ubckpjwm5dvfgvepd6h5h7slvapt77yd6dtq47m4kod7bles.py
# Source Nodes: [l__mod___blocks_1_norm_mlp, patch_embed_9], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_1_norm_mlp => add_38, add_39, mul_35, mul_36, rsqrt_11, sub_15, var_mean_11
# patch_embed_9 => add_37
triton_per_fused_add_native_layer_norm_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36ouu5emugogmmutljfpzuwcrjt22qleec3uqbnrhvkoevjfi7a.py
# Source Nodes: [patch_embed_11, patch_embed_9], Original ATen: [aten.add]
# patch_embed_11 => add_41
# patch_embed_9 => add_37
triton_poi_fused_add_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clxb5f6fc45rfd7vn2yaaxdz3q4celwjdriip6unuexigf2ufrqj.py
# Source Nodes: [l__mod___blocks_11_norm1_proj, pixel_embed_36], Original ATen: [aten.add, aten.native_layer_norm]
# l__mod___blocks_11_norm1_proj => add_202, add_203, clone_177, mul_210, mul_211, rsqrt_59, sub_82, var_mean_59
# pixel_embed_36 => add_201
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 24.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkm7qmiwgelxqp325vrmiv3bkbh2uot7qg4jlsnf3xxnefdpakz.py
# Source Nodes: [patch_embed_49, patch_embed_51, x_221], Original ATen: [aten.add, aten.native_layer_norm]
# patch_embed_49 => add_207
# patch_embed_51 => add_211
# x_221 => var_mean_62
triton_per_fused_add_native_layer_norm_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4lsovfrjhoqarwc5biwtxyutkglks7bckh6guz4gqgg3q5emgc.py
# Source Nodes: [x_223], Original ATen: [aten.clone]
# x_223 => clone_184
triton_poi_fused_clone_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (75648*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (197*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (197*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 24, 4, 4), (384, 16, 4, 1))
    assert_size_stride(arg1_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg2_1, (1, 197, 384), (75648, 384, 1))
    assert_size_stride(arg3_1, (24, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg4_1, (24, ), (1, ))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (384, ), (1, ))
    assert_size_stride(arg7_1, (384, 384), (384, 1))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (384, ), (1, ))
    assert_size_stride(arg11_1, (24, ), (1, ))
    assert_size_stride(arg12_1, (24, ), (1, ))
    assert_size_stride(arg13_1, (48, 24), (24, 1))
    assert_size_stride(arg14_1, (24, 24), (24, 1))
    assert_size_stride(arg15_1, (24, 24), (24, 1))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (24, ), (1, ))
    assert_size_stride(arg19_1, (96, 24), (24, 1))
    assert_size_stride(arg20_1, (96, ), (1, ))
    assert_size_stride(arg21_1, (24, 96), (96, 1))
    assert_size_stride(arg22_1, (24, ), (1, ))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (384, 384), (384, 1))
    assert_size_stride(arg26_1, (384, ), (1, ))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (768, 384), (384, 1))
    assert_size_stride(arg30_1, (384, 384), (384, 1))
    assert_size_stride(arg31_1, (384, 384), (384, 1))
    assert_size_stride(arg32_1, (384, ), (1, ))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (384, ), (1, ))
    assert_size_stride(arg35_1, (1536, 384), (384, 1))
    assert_size_stride(arg36_1, (1536, ), (1, ))
    assert_size_stride(arg37_1, (384, 1536), (1536, 1))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (24, ), (1, ))
    assert_size_stride(arg40_1, (24, ), (1, ))
    assert_size_stride(arg41_1, (48, 24), (24, 1))
    assert_size_stride(arg42_1, (24, 24), (24, 1))
    assert_size_stride(arg43_1, (24, 24), (24, 1))
    assert_size_stride(arg44_1, (24, ), (1, ))
    assert_size_stride(arg45_1, (24, ), (1, ))
    assert_size_stride(arg46_1, (24, ), (1, ))
    assert_size_stride(arg47_1, (96, 24), (24, 1))
    assert_size_stride(arg48_1, (96, ), (1, ))
    assert_size_stride(arg49_1, (24, 96), (96, 1))
    assert_size_stride(arg50_1, (24, ), (1, ))
    assert_size_stride(arg51_1, (24, ), (1, ))
    assert_size_stride(arg52_1, (24, ), (1, ))
    assert_size_stride(arg53_1, (384, 384), (384, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (384, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (768, 384), (384, 1))
    assert_size_stride(arg58_1, (384, 384), (384, 1))
    assert_size_stride(arg59_1, (384, 384), (384, 1))
    assert_size_stride(arg60_1, (384, ), (1, ))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (1536, 384), (384, 1))
    assert_size_stride(arg64_1, (1536, ), (1, ))
    assert_size_stride(arg65_1, (384, 1536), (1536, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (24, ), (1, ))
    assert_size_stride(arg68_1, (24, ), (1, ))
    assert_size_stride(arg69_1, (48, 24), (24, 1))
    assert_size_stride(arg70_1, (24, 24), (24, 1))
    assert_size_stride(arg71_1, (24, 24), (24, 1))
    assert_size_stride(arg72_1, (24, ), (1, ))
    assert_size_stride(arg73_1, (24, ), (1, ))
    assert_size_stride(arg74_1, (24, ), (1, ))
    assert_size_stride(arg75_1, (96, 24), (24, 1))
    assert_size_stride(arg76_1, (96, ), (1, ))
    assert_size_stride(arg77_1, (24, 96), (96, 1))
    assert_size_stride(arg78_1, (24, ), (1, ))
    assert_size_stride(arg79_1, (24, ), (1, ))
    assert_size_stride(arg80_1, (24, ), (1, ))
    assert_size_stride(arg81_1, (384, 384), (384, 1))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (768, 384), (384, 1))
    assert_size_stride(arg86_1, (384, 384), (384, 1))
    assert_size_stride(arg87_1, (384, 384), (384, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (1536, 384), (384, 1))
    assert_size_stride(arg92_1, (1536, ), (1, ))
    assert_size_stride(arg93_1, (384, 1536), (1536, 1))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (24, ), (1, ))
    assert_size_stride(arg96_1, (24, ), (1, ))
    assert_size_stride(arg97_1, (48, 24), (24, 1))
    assert_size_stride(arg98_1, (24, 24), (24, 1))
    assert_size_stride(arg99_1, (24, 24), (24, 1))
    assert_size_stride(arg100_1, (24, ), (1, ))
    assert_size_stride(arg101_1, (24, ), (1, ))
    assert_size_stride(arg102_1, (24, ), (1, ))
    assert_size_stride(arg103_1, (96, 24), (24, 1))
    assert_size_stride(arg104_1, (96, ), (1, ))
    assert_size_stride(arg105_1, (24, 96), (96, 1))
    assert_size_stride(arg106_1, (24, ), (1, ))
    assert_size_stride(arg107_1, (24, ), (1, ))
    assert_size_stride(arg108_1, (24, ), (1, ))
    assert_size_stride(arg109_1, (384, 384), (384, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (768, 384), (384, 1))
    assert_size_stride(arg114_1, (384, 384), (384, 1))
    assert_size_stride(arg115_1, (384, 384), (384, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (384, ), (1, ))
    assert_size_stride(arg119_1, (1536, 384), (384, 1))
    assert_size_stride(arg120_1, (1536, ), (1, ))
    assert_size_stride(arg121_1, (384, 1536), (1536, 1))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (24, ), (1, ))
    assert_size_stride(arg124_1, (24, ), (1, ))
    assert_size_stride(arg125_1, (48, 24), (24, 1))
    assert_size_stride(arg126_1, (24, 24), (24, 1))
    assert_size_stride(arg127_1, (24, 24), (24, 1))
    assert_size_stride(arg128_1, (24, ), (1, ))
    assert_size_stride(arg129_1, (24, ), (1, ))
    assert_size_stride(arg130_1, (24, ), (1, ))
    assert_size_stride(arg131_1, (96, 24), (24, 1))
    assert_size_stride(arg132_1, (96, ), (1, ))
    assert_size_stride(arg133_1, (24, 96), (96, 1))
    assert_size_stride(arg134_1, (24, ), (1, ))
    assert_size_stride(arg135_1, (24, ), (1, ))
    assert_size_stride(arg136_1, (24, ), (1, ))
    assert_size_stride(arg137_1, (384, 384), (384, 1))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (768, 384), (384, 1))
    assert_size_stride(arg142_1, (384, 384), (384, 1))
    assert_size_stride(arg143_1, (384, 384), (384, 1))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (1536, 384), (384, 1))
    assert_size_stride(arg148_1, (1536, ), (1, ))
    assert_size_stride(arg149_1, (384, 1536), (1536, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (24, ), (1, ))
    assert_size_stride(arg152_1, (24, ), (1, ))
    assert_size_stride(arg153_1, (48, 24), (24, 1))
    assert_size_stride(arg154_1, (24, 24), (24, 1))
    assert_size_stride(arg155_1, (24, 24), (24, 1))
    assert_size_stride(arg156_1, (24, ), (1, ))
    assert_size_stride(arg157_1, (24, ), (1, ))
    assert_size_stride(arg158_1, (24, ), (1, ))
    assert_size_stride(arg159_1, (96, 24), (24, 1))
    assert_size_stride(arg160_1, (96, ), (1, ))
    assert_size_stride(arg161_1, (24, 96), (96, 1))
    assert_size_stride(arg162_1, (24, ), (1, ))
    assert_size_stride(arg163_1, (24, ), (1, ))
    assert_size_stride(arg164_1, (24, ), (1, ))
    assert_size_stride(arg165_1, (384, 384), (384, 1))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (768, 384), (384, 1))
    assert_size_stride(arg170_1, (384, 384), (384, 1))
    assert_size_stride(arg171_1, (384, 384), (384, 1))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (1536, 384), (384, 1))
    assert_size_stride(arg176_1, (1536, ), (1, ))
    assert_size_stride(arg177_1, (384, 1536), (1536, 1))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (24, ), (1, ))
    assert_size_stride(arg180_1, (24, ), (1, ))
    assert_size_stride(arg181_1, (48, 24), (24, 1))
    assert_size_stride(arg182_1, (24, 24), (24, 1))
    assert_size_stride(arg183_1, (24, 24), (24, 1))
    assert_size_stride(arg184_1, (24, ), (1, ))
    assert_size_stride(arg185_1, (24, ), (1, ))
    assert_size_stride(arg186_1, (24, ), (1, ))
    assert_size_stride(arg187_1, (96, 24), (24, 1))
    assert_size_stride(arg188_1, (96, ), (1, ))
    assert_size_stride(arg189_1, (24, 96), (96, 1))
    assert_size_stride(arg190_1, (24, ), (1, ))
    assert_size_stride(arg191_1, (24, ), (1, ))
    assert_size_stride(arg192_1, (24, ), (1, ))
    assert_size_stride(arg193_1, (384, 384), (384, 1))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (768, 384), (384, 1))
    assert_size_stride(arg198_1, (384, 384), (384, 1))
    assert_size_stride(arg199_1, (384, 384), (384, 1))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (384, ), (1, ))
    assert_size_stride(arg203_1, (1536, 384), (384, 1))
    assert_size_stride(arg204_1, (1536, ), (1, ))
    assert_size_stride(arg205_1, (384, 1536), (1536, 1))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (24, ), (1, ))
    assert_size_stride(arg208_1, (24, ), (1, ))
    assert_size_stride(arg209_1, (48, 24), (24, 1))
    assert_size_stride(arg210_1, (24, 24), (24, 1))
    assert_size_stride(arg211_1, (24, 24), (24, 1))
    assert_size_stride(arg212_1, (24, ), (1, ))
    assert_size_stride(arg213_1, (24, ), (1, ))
    assert_size_stride(arg214_1, (24, ), (1, ))
    assert_size_stride(arg215_1, (96, 24), (24, 1))
    assert_size_stride(arg216_1, (96, ), (1, ))
    assert_size_stride(arg217_1, (24, 96), (96, 1))
    assert_size_stride(arg218_1, (24, ), (1, ))
    assert_size_stride(arg219_1, (24, ), (1, ))
    assert_size_stride(arg220_1, (24, ), (1, ))
    assert_size_stride(arg221_1, (384, 384), (384, 1))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (384, ), (1, ))
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (768, 384), (384, 1))
    assert_size_stride(arg226_1, (384, 384), (384, 1))
    assert_size_stride(arg227_1, (384, 384), (384, 1))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (1536, 384), (384, 1))
    assert_size_stride(arg232_1, (1536, ), (1, ))
    assert_size_stride(arg233_1, (384, 1536), (1536, 1))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (24, ), (1, ))
    assert_size_stride(arg236_1, (24, ), (1, ))
    assert_size_stride(arg237_1, (48, 24), (24, 1))
    assert_size_stride(arg238_1, (24, 24), (24, 1))
    assert_size_stride(arg239_1, (24, 24), (24, 1))
    assert_size_stride(arg240_1, (24, ), (1, ))
    assert_size_stride(arg241_1, (24, ), (1, ))
    assert_size_stride(arg242_1, (24, ), (1, ))
    assert_size_stride(arg243_1, (96, 24), (24, 1))
    assert_size_stride(arg244_1, (96, ), (1, ))
    assert_size_stride(arg245_1, (24, 96), (96, 1))
    assert_size_stride(arg246_1, (24, ), (1, ))
    assert_size_stride(arg247_1, (24, ), (1, ))
    assert_size_stride(arg248_1, (24, ), (1, ))
    assert_size_stride(arg249_1, (384, 384), (384, 1))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (768, 384), (384, 1))
    assert_size_stride(arg254_1, (384, 384), (384, 1))
    assert_size_stride(arg255_1, (384, 384), (384, 1))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (1536, 384), (384, 1))
    assert_size_stride(arg260_1, (1536, ), (1, ))
    assert_size_stride(arg261_1, (384, 1536), (1536, 1))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (24, ), (1, ))
    assert_size_stride(arg264_1, (24, ), (1, ))
    assert_size_stride(arg265_1, (48, 24), (24, 1))
    assert_size_stride(arg266_1, (24, 24), (24, 1))
    assert_size_stride(arg267_1, (24, 24), (24, 1))
    assert_size_stride(arg268_1, (24, ), (1, ))
    assert_size_stride(arg269_1, (24, ), (1, ))
    assert_size_stride(arg270_1, (24, ), (1, ))
    assert_size_stride(arg271_1, (96, 24), (24, 1))
    assert_size_stride(arg272_1, (96, ), (1, ))
    assert_size_stride(arg273_1, (24, 96), (96, 1))
    assert_size_stride(arg274_1, (24, ), (1, ))
    assert_size_stride(arg275_1, (24, ), (1, ))
    assert_size_stride(arg276_1, (24, ), (1, ))
    assert_size_stride(arg277_1, (384, 384), (384, 1))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (768, 384), (384, 1))
    assert_size_stride(arg282_1, (384, 384), (384, 1))
    assert_size_stride(arg283_1, (384, 384), (384, 1))
    assert_size_stride(arg284_1, (384, ), (1, ))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (384, ), (1, ))
    assert_size_stride(arg287_1, (1536, 384), (384, 1))
    assert_size_stride(arg288_1, (1536, ), (1, ))
    assert_size_stride(arg289_1, (384, 1536), (1536, 1))
    assert_size_stride(arg290_1, (384, ), (1, ))
    assert_size_stride(arg291_1, (24, ), (1, ))
    assert_size_stride(arg292_1, (24, ), (1, ))
    assert_size_stride(arg293_1, (48, 24), (24, 1))
    assert_size_stride(arg294_1, (24, 24), (24, 1))
    assert_size_stride(arg295_1, (24, 24), (24, 1))
    assert_size_stride(arg296_1, (24, ), (1, ))
    assert_size_stride(arg297_1, (24, ), (1, ))
    assert_size_stride(arg298_1, (24, ), (1, ))
    assert_size_stride(arg299_1, (96, 24), (24, 1))
    assert_size_stride(arg300_1, (96, ), (1, ))
    assert_size_stride(arg301_1, (24, 96), (96, 1))
    assert_size_stride(arg302_1, (24, ), (1, ))
    assert_size_stride(arg303_1, (24, ), (1, ))
    assert_size_stride(arg304_1, (24, ), (1, ))
    assert_size_stride(arg305_1, (384, 384), (384, 1))
    assert_size_stride(arg306_1, (384, ), (1, ))
    assert_size_stride(arg307_1, (384, ), (1, ))
    assert_size_stride(arg308_1, (384, ), (1, ))
    assert_size_stride(arg309_1, (768, 384), (384, 1))
    assert_size_stride(arg310_1, (384, 384), (384, 1))
    assert_size_stride(arg311_1, (384, 384), (384, 1))
    assert_size_stride(arg312_1, (384, ), (1, ))
    assert_size_stride(arg313_1, (384, ), (1, ))
    assert_size_stride(arg314_1, (384, ), (1, ))
    assert_size_stride(arg315_1, (1536, 384), (384, 1))
    assert_size_stride(arg316_1, (1536, ), (1, ))
    assert_size_stride(arg317_1, (384, 1536), (1536, 1))
    assert_size_stride(arg318_1, (384, ), (1, ))
    assert_size_stride(arg319_1, (24, ), (1, ))
    assert_size_stride(arg320_1, (24, ), (1, ))
    assert_size_stride(arg321_1, (48, 24), (24, 1))
    assert_size_stride(arg322_1, (24, 24), (24, 1))
    assert_size_stride(arg323_1, (24, 24), (24, 1))
    assert_size_stride(arg324_1, (24, ), (1, ))
    assert_size_stride(arg325_1, (24, ), (1, ))
    assert_size_stride(arg326_1, (24, ), (1, ))
    assert_size_stride(arg327_1, (96, 24), (24, 1))
    assert_size_stride(arg328_1, (96, ), (1, ))
    assert_size_stride(arg329_1, (24, 96), (96, 1))
    assert_size_stride(arg330_1, (24, ), (1, ))
    assert_size_stride(arg331_1, (24, ), (1, ))
    assert_size_stride(arg332_1, (24, ), (1, ))
    assert_size_stride(arg333_1, (384, 384), (384, 1))
    assert_size_stride(arg334_1, (384, ), (1, ))
    assert_size_stride(arg335_1, (384, ), (1, ))
    assert_size_stride(arg336_1, (384, ), (1, ))
    assert_size_stride(arg337_1, (768, 384), (384, 1))
    assert_size_stride(arg338_1, (384, 384), (384, 1))
    assert_size_stride(arg339_1, (384, 384), (384, 1))
    assert_size_stride(arg340_1, (384, ), (1, ))
    assert_size_stride(arg341_1, (384, ), (1, ))
    assert_size_stride(arg342_1, (384, ), (1, ))
    assert_size_stride(arg343_1, (1536, 384), (384, 1))
    assert_size_stride(arg344_1, (1536, ), (1, ))
    assert_size_stride(arg345_1, (384, 1536), (1536, 1))
    assert_size_stride(arg346_1, (384, ), (1, ))
    assert_size_stride(arg347_1, (384, ), (1, ))
    assert_size_stride(arg348_1, (384, ), (1, ))
    assert_size_stride(arg349_1, (1000, 384), (384, 1))
    assert_size_stride(arg350_1, (1000, ), (1, ))
    assert_size_stride(arg351_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg351_1, arg3_1, stride=(4, 4), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg351_1
        del arg3_1
        buf1 = empty_strided((8, 196, 1, 3), (588, 3, 4704, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 3), (588, 3, 4704, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 3), (588, 3, 4704, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg4_1, arg0_1, buf1, buf2, buf3, 4704, 128, grid=grid(4704), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 3, grid=grid(1568), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_2.run(buf0, arg4_1, arg0_1, buf4, buf5, arg5_1, arg6_1, buf7, 602112, grid=grid(602112), stream=stream0)
        del arg5_1
        del arg6_1
        buf8 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___proj], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf7, (1568, 384), (384, 1), 0), reinterpret_tensor(arg7_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf8)
        del arg7_1
        del arg8_1
        buf9 = buf5; del buf5  # reuse
        buf10 = buf4; del buf4  # reuse
        # Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf8, buf9, buf10, 1568, 384, grid=grid(1568), stream=stream0)
        buf12 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_4.run(buf0, arg4_1, arg0_1, buf12, buf13, 25088, 24, grid=grid(25088), stream=stream0)
        buf15 = reinterpret_tensor(buf7, (1568, 16, 24), (384, 24, 1), 0); del buf7  # reuse
        # Source Nodes: [l__mod___blocks_0_norm_in], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_5.run(buf0, arg4_1, arg0_1, buf12, buf13, arg11_1, arg12_1, buf15, 37632, 16, grid=grid(37632, 16), stream=stream0)
        del arg11_1
        del arg12_1
        del buf12
        del buf13
        buf16 = empty((25088, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (25088, 24), (24, 1), 0), reinterpret_tensor(arg13_1, (24, 48), (1, 24), 0), out=buf16)
        del arg13_1
        buf17 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf16, buf17, 602112, grid=grid(602112), stream=stream0)
        buf18 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf16, buf18, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf19 = empty((6272, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf17, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf18, (6272, 6, 16), (96, 16, 1), 0), out=buf19)
        buf23 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf19, buf23, 100352, 16, grid=grid(100352), stream=stream0)
        buf22 = reinterpret_tensor(buf18, (25088, 24), (24, 1), 0); del buf18  # reuse
        # Source Nodes: [l__mod___blocks_0_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (25088, 24), (24, 1), 0), reinterpret_tensor(arg14_1, (24, 24), (1, 24), 0), out=buf22)
        del arg14_1
        buf24 = reinterpret_tensor(buf15, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf15  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf22, buf24, 602112, grid=grid(602112), stream=stream0)
        buf25 = reinterpret_tensor(buf22, (6272, 16, 6), (96, 6, 1), 0); del buf22  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf24, (6272, 16, 6), (96, 6, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf24, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf24  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf25, buf26, 602112, grid=grid(602112), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (25088, 24), (24, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (25088, 24), (24, 1), 0), reinterpret_tensor(arg15_1, (24, 24), (1, 24), 0), out=buf27)
        del arg15_1
        buf28 = reinterpret_tensor(buf27, (1568, 16, 24), (384, 24, 1), 0); del buf27  # reuse
        buf32 = reinterpret_tensor(buf26, (1568, 16, 24), (384, 24, 1), 0); del buf26  # reuse
        # Source Nodes: [l__mod___blocks_0_norm_mlp_in, pixel_embed_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf28, buf0, arg4_1, arg0_1, arg16_1, arg17_1, arg18_1, buf32, 25088, 24, grid=grid(25088), stream=stream0)
        del arg0_1
        del arg16_1
        del arg17_1
        del arg18_1
        del arg4_1
        buf33 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (25088, 24), (24, 1), 0), reinterpret_tensor(arg19_1, (24, 96), (1, 24), 0), out=buf33)
        del arg19_1
        buf34 = reinterpret_tensor(buf33, (1568, 16, 96), (1536, 96, 1), 0); del buf33  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf34, arg20_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg20_1
        buf35 = reinterpret_tensor(buf32, (25088, 24), (24, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (25088, 96), (96, 1), 0), reinterpret_tensor(arg21_1, (96, 24), (1, 96), 0), out=buf35)
        del arg21_1
        buf40 = reinterpret_tensor(buf0, (1568, 16, 24), (384, 24, 1), 0); del buf0  # reuse
        buf65 = reinterpret_tensor(buf17, (1568, 16, 24), (384, 24, 1), 0); del buf17  # reuse
        # Source Nodes: [l__mod___blocks_0_norm1_proj, l__mod___blocks_1_norm_in, pixel_embed_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf28, buf35, arg22_1, arg23_1, arg24_1, arg39_1, arg40_1, buf40, buf65, 25088, 24, grid=grid(25088), stream=stream0)
        del arg23_1
        del arg24_1
        del arg39_1
        del arg40_1
        buf39 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25, patch_embed_2], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_14.run(arg1_1, buf8, buf9, buf10, arg9_1, arg10_1, arg2_1, buf39, 605184, grid=grid(605184), stream=stream0)
        del arg10_1
        del arg1_1
        del arg2_1
        del arg9_1
        del buf10
        del buf9
        buf41 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (1568, 384), (384, 1), 0), reinterpret_tensor(arg25_1, (384, 384), (1, 384), 0), out=buf41)
        del arg25_1
        buf45 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24, l__mod___blocks_0_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf39, buf41, arg26_1, arg27_1, arg28_1, buf45, 1576, 384, grid=grid(1576), stream=stream0)
        del arg27_1
        del arg28_1
        buf46 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1576, 384), (384, 1), 0), reinterpret_tensor(arg29_1, (384, 768), (1, 384), 0), out=buf46)
        del arg29_1
        buf47 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf46, buf47, 605184, grid=grid(605184), stream=stream0)
        buf48 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf46, buf48, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf49 = empty((48, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf48, (48, 64, 197), (12608, 197, 1), 0), out=buf49)
        buf53 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf49, buf53, 9456, 197, grid=grid(9456), stream=stream0)
        buf52 = reinterpret_tensor(buf48, (1576, 384), (384, 1), 0); del buf48  # reuse
        # Source Nodes: [l__mod___blocks_0_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1576, 384), (384, 1), 0), reinterpret_tensor(arg30_1, (384, 384), (1, 384), 0), out=buf52)
        del arg30_1
        buf54 = reinterpret_tensor(buf45, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf45  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf52, buf54, 605184, grid=grid(605184), stream=stream0)
        buf55 = reinterpret_tensor(buf52, (48, 197, 64), (12608, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf54, (48, 197, 64), (12608, 64, 1), 0), out=buf55)
        buf56 = reinterpret_tensor(buf54, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf55, buf56, 605184, grid=grid(605184), stream=stream0)
        buf57 = reinterpret_tensor(buf55, (1576, 384), (384, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (1576, 384), (384, 1), 0), reinterpret_tensor(arg31_1, (384, 384), (1, 384), 0), out=buf57)
        del arg31_1
        buf58 = reinterpret_tensor(buf57, (8, 197, 384), (75648, 384, 1), 0); del buf57  # reuse
        buf89 = reinterpret_tensor(buf56, (8, 197, 384), (75648, 384, 1), 0); del buf56  # reuse
        # Source Nodes: [cat_24, l__mod___blocks_0_norm_mlp, patch_embed_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf58, buf39, buf41, arg26_1, arg32_1, arg33_1, arg34_1, buf89, 1576, 384, grid=grid(1576), stream=stream0)
        del arg26_1
        del arg32_1
        del arg33_1
        del arg34_1
        buf66 = buf16; del buf16  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (25088, 24), (24, 1), 0), reinterpret_tensor(arg41_1, (24, 48), (1, 24), 0), out=buf66)
        del arg41_1
        buf67 = reinterpret_tensor(buf41, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf41  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf66, buf67, 602112, grid=grid(602112), stream=stream0)
        buf68 = reinterpret_tensor(buf40, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf40  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf66, buf68, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf69 = reinterpret_tensor(buf23, (6272, 16, 16), (256, 16, 1), 0); del buf23  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf67, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf68, (6272, 6, 16), (96, 16, 1), 0), out=buf69)
        buf73 = reinterpret_tensor(buf19, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf19  # reuse
        # Source Nodes: [attn_6, attn_7], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf69, buf73, 100352, 16, grid=grid(100352), stream=stream0)
        buf72 = reinterpret_tensor(buf68, (25088, 24), (24, 1), 0); del buf68  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (25088, 24), (24, 1), 0), reinterpret_tensor(arg42_1, (24, 24), (1, 24), 0), out=buf72)
        del arg42_1
        buf74 = reinterpret_tensor(buf65, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf65  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf72, buf74, 602112, grid=grid(602112), stream=stream0)
        buf75 = reinterpret_tensor(buf72, (6272, 16, 6), (96, 6, 1), 0); del buf72  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf74, (6272, 16, 6), (96, 6, 1), 0), out=buf75)
        buf76 = reinterpret_tensor(buf74, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf74  # reuse
        # Source Nodes: [x_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf75, buf76, 602112, grid=grid(602112), stream=stream0)
        buf77 = reinterpret_tensor(buf75, (25088, 24), (24, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (25088, 24), (24, 1), 0), reinterpret_tensor(arg43_1, (24, 24), (1, 24), 0), out=buf77)
        del arg43_1
        buf78 = reinterpret_tensor(buf77, (1568, 16, 24), (384, 24, 1), 0); del buf77  # reuse
        buf82 = reinterpret_tensor(buf76, (1568, 16, 24), (384, 24, 1), 0); del buf76  # reuse
        # Source Nodes: [l__mod___blocks_1_norm_mlp_in, pixel_embed_3, pixel_embed_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf78, buf28, buf35, arg22_1, arg44_1, arg45_1, arg46_1, buf82, 25088, 24, grid=grid(25088), stream=stream0)
        del arg22_1
        del arg44_1
        del arg45_1
        del arg46_1
        buf83 = reinterpret_tensor(buf34, (25088, 96), (96, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (25088, 24), (24, 1), 0), reinterpret_tensor(arg47_1, (24, 96), (1, 24), 0), out=buf83)
        del arg47_1
        buf84 = reinterpret_tensor(buf83, (1568, 16, 96), (1536, 96, 1), 0); del buf83  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf84, arg48_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg48_1
        buf85 = reinterpret_tensor(buf82, (25088, 24), (24, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (25088, 96), (96, 1), 0), reinterpret_tensor(arg49_1, (96, 24), (1, 96), 0), out=buf85)
        del arg49_1
        buf93 = reinterpret_tensor(buf35, (1568, 16, 24), (384, 24, 1), 0); del buf35  # reuse
        buf118 = buf28; del buf28  # reuse
        # Source Nodes: [l__mod___blocks_1_norm1_proj, l__mod___blocks_2_norm_in, pixel_embed_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf78, buf85, arg50_1, arg51_1, arg52_1, arg67_1, arg68_1, buf93, buf118, 25088, 24, grid=grid(25088), stream=stream0)
        del arg51_1
        del arg52_1
        del arg67_1
        del arg68_1
        buf90 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf89, (1576, 384), (384, 1), 0), reinterpret_tensor(arg35_1, (384, 1536), (1, 384), 0), out=buf90)
        del arg35_1
        buf91 = reinterpret_tensor(buf90, (8, 197, 1536), (302592, 1536, 1), 0); del buf90  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf91, arg36_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg36_1
        buf92 = reinterpret_tensor(buf89, (1576, 384), (384, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg37_1, (1536, 384), (1, 1536), 0), out=buf92)
        del arg37_1
        buf94 = reinterpret_tensor(buf67, (1568, 384), (384, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (1568, 384), (384, 1), 0), reinterpret_tensor(arg53_1, (384, 384), (1, 384), 0), out=buf94)
        del arg53_1
        buf95 = reinterpret_tensor(buf92, (8, 197, 384), (75648, 384, 1), 0); del buf92  # reuse
        buf99 = buf39; del buf39  # reuse
        # Source Nodes: [cat_23, l__mod___blocks_1_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf95, buf58, arg38_1, buf94, arg54_1, arg55_1, arg56_1, buf99, 1576, 384, grid=grid(1576), stream=stream0)
        del arg38_1
        del arg54_1
        del arg55_1
        del arg56_1
        buf100 = buf46; del buf46  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (1576, 384), (384, 1), 0), reinterpret_tensor(arg57_1, (384, 768), (1, 384), 0), out=buf100)
        del arg57_1
        buf101 = reinterpret_tensor(buf58, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf58  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf100, buf101, 605184, grid=grid(605184), stream=stream0)
        buf102 = reinterpret_tensor(buf47, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf47  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf100, buf102, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf103 = reinterpret_tensor(buf53, (48, 197, 197), (38809, 197, 1), 0); del buf53  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf102, (48, 64, 197), (12608, 197, 1), 0), out=buf103)
        buf107 = reinterpret_tensor(buf49, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf49  # reuse
        # Source Nodes: [attn_10, attn_9], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf103, buf107, 9456, 197, grid=grid(9456), stream=stream0)
        buf106 = reinterpret_tensor(buf102, (1576, 384), (384, 1), 0); del buf102  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (1576, 384), (384, 1), 0), reinterpret_tensor(arg58_1, (384, 384), (1, 384), 0), out=buf106)
        del arg58_1
        buf108 = reinterpret_tensor(buf99, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf99  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf106, buf108, 605184, grid=grid(605184), stream=stream0)
        buf109 = reinterpret_tensor(buf106, (48, 197, 64), (12608, 64, 1), 0); del buf106  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf108, (48, 197, 64), (12608, 64, 1), 0), out=buf109)
        buf110 = reinterpret_tensor(buf108, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [x_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf109, buf110, 605184, grid=grid(605184), stream=stream0)
        buf111 = reinterpret_tensor(buf109, (1576, 384), (384, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (1576, 384), (384, 1), 0), reinterpret_tensor(arg59_1, (384, 384), (1, 384), 0), out=buf111)
        del arg59_1
        buf142 = reinterpret_tensor(buf110, (8, 197, 384), (75648, 384, 1), 0); del buf110  # reuse
        # Source Nodes: [l__mod___blocks_1_norm_mlp, patch_embed_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf95, buf111, arg60_1, arg61_1, arg62_1, buf142, 1576, 384, grid=grid(1576), stream=stream0)
        del arg61_1
        del arg62_1
        buf119 = buf66; del buf66  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (25088, 24), (24, 1), 0), reinterpret_tensor(arg69_1, (24, 48), (1, 24), 0), out=buf119)
        del arg69_1
        buf120 = reinterpret_tensor(buf94, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf94  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf119, buf120, 602112, grid=grid(602112), stream=stream0)
        buf121 = reinterpret_tensor(buf93, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf93  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf119, buf121, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf122 = reinterpret_tensor(buf73, (6272, 16, 16), (256, 16, 1), 0); del buf73  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf121, (6272, 6, 16), (96, 16, 1), 0), out=buf122)
        buf126 = reinterpret_tensor(buf69, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf69  # reuse
        # Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf122, buf126, 100352, 16, grid=grid(100352), stream=stream0)
        buf125 = reinterpret_tensor(buf121, (25088, 24), (24, 1), 0); del buf121  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (25088, 24), (24, 1), 0), reinterpret_tensor(arg70_1, (24, 24), (1, 24), 0), out=buf125)
        del arg70_1
        buf127 = reinterpret_tensor(buf118, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf118  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf125, buf127, 602112, grid=grid(602112), stream=stream0)
        buf128 = reinterpret_tensor(buf125, (6272, 16, 6), (96, 6, 1), 0); del buf125  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf126, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf127, (6272, 16, 6), (96, 6, 1), 0), out=buf128)
        buf129 = reinterpret_tensor(buf127, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf127  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf128, buf129, 602112, grid=grid(602112), stream=stream0)
        buf130 = reinterpret_tensor(buf128, (25088, 24), (24, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (25088, 24), (24, 1), 0), reinterpret_tensor(arg71_1, (24, 24), (1, 24), 0), out=buf130)
        del arg71_1
        buf131 = reinterpret_tensor(buf130, (1568, 16, 24), (384, 24, 1), 0); del buf130  # reuse
        buf135 = reinterpret_tensor(buf129, (1568, 16, 24), (384, 24, 1), 0); del buf129  # reuse
        # Source Nodes: [l__mod___blocks_2_norm_mlp_in, pixel_embed_6, pixel_embed_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf131, buf78, buf85, arg50_1, arg72_1, arg73_1, arg74_1, buf135, 25088, 24, grid=grid(25088), stream=stream0)
        del arg50_1
        del arg72_1
        del arg73_1
        del arg74_1
        buf136 = reinterpret_tensor(buf84, (25088, 96), (96, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (25088, 24), (24, 1), 0), reinterpret_tensor(arg75_1, (24, 96), (1, 24), 0), out=buf136)
        del arg75_1
        buf137 = reinterpret_tensor(buf136, (1568, 16, 96), (1536, 96, 1), 0); del buf136  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf137, arg76_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg76_1
        buf138 = reinterpret_tensor(buf135, (25088, 24), (24, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (25088, 96), (96, 1), 0), reinterpret_tensor(arg77_1, (96, 24), (1, 96), 0), out=buf138)
        del arg77_1
        buf147 = reinterpret_tensor(buf85, (1568, 16, 24), (384, 24, 1), 0); del buf85  # reuse
        buf172 = buf78; del buf78  # reuse
        # Source Nodes: [l__mod___blocks_2_norm1_proj, l__mod___blocks_3_norm_in, pixel_embed_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf131, buf138, arg78_1, arg79_1, arg80_1, arg95_1, arg96_1, buf147, buf172, 25088, 24, grid=grid(25088), stream=stream0)
        del arg79_1
        del arg80_1
        del arg95_1
        del arg96_1
        buf143 = reinterpret_tensor(buf91, (1576, 1536), (1536, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1576, 384), (384, 1), 0), reinterpret_tensor(arg63_1, (384, 1536), (1, 384), 0), out=buf143)
        del arg63_1
        buf144 = reinterpret_tensor(buf143, (8, 197, 1536), (302592, 1536, 1), 0); del buf143  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf144, arg64_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg64_1
        buf145 = reinterpret_tensor(buf142, (1576, 384), (384, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg65_1, (1536, 384), (1, 1536), 0), out=buf145)
        del arg65_1
        buf146 = reinterpret_tensor(buf145, (8, 197, 384), (75648, 384, 1), 0); del buf145  # reuse
        # Source Nodes: [patch_embed_11, patch_embed_9], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf146, buf95, buf111, arg60_1, arg66_1, 605184, grid=grid(605184), stream=stream0)
        del arg60_1
        del arg66_1
        buf148 = reinterpret_tensor(buf120, (1568, 384), (384, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1568, 384), (384, 1), 0), reinterpret_tensor(arg81_1, (384, 384), (1, 384), 0), out=buf148)
        del arg81_1
        buf152 = buf95; del buf95  # reuse
        # Source Nodes: [cat_22, l__mod___blocks_2_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf146, buf148, arg82_1, arg83_1, arg84_1, buf152, 1576, 384, grid=grid(1576), stream=stream0)
        del arg83_1
        del arg84_1
        buf153 = buf100; del buf100  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1576, 384), (384, 1), 0), reinterpret_tensor(arg85_1, (384, 768), (1, 384), 0), out=buf153)
        del arg85_1
        buf154 = reinterpret_tensor(buf111, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf111  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf153, buf154, 605184, grid=grid(605184), stream=stream0)
        buf155 = reinterpret_tensor(buf101, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf101  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf153, buf155, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf156 = reinterpret_tensor(buf107, (48, 197, 197), (38809, 197, 1), 0); del buf107  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf155, (48, 64, 197), (12608, 197, 1), 0), out=buf156)
        buf160 = reinterpret_tensor(buf103, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf103  # reuse
        # Source Nodes: [attn_15, attn_16], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf156, buf160, 9456, 197, grid=grid(9456), stream=stream0)
        buf159 = reinterpret_tensor(buf155, (1576, 384), (384, 1), 0); del buf155  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1576, 384), (384, 1), 0), reinterpret_tensor(arg86_1, (384, 384), (1, 384), 0), out=buf159)
        del arg86_1
        buf161 = reinterpret_tensor(buf152, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf152  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf159, buf161, 605184, grid=grid(605184), stream=stream0)
        buf162 = reinterpret_tensor(buf159, (48, 197, 64), (12608, 64, 1), 0); del buf159  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf161, (48, 197, 64), (12608, 64, 1), 0), out=buf162)
        buf163 = reinterpret_tensor(buf161, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf161  # reuse
        # Source Nodes: [x_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf162, buf163, 605184, grid=grid(605184), stream=stream0)
        buf164 = reinterpret_tensor(buf162, (1576, 384), (384, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf163, (1576, 384), (384, 1), 0), reinterpret_tensor(arg87_1, (384, 384), (1, 384), 0), out=buf164)
        del arg87_1
        buf165 = reinterpret_tensor(buf164, (8, 197, 384), (75648, 384, 1), 0); del buf164  # reuse
        buf196 = reinterpret_tensor(buf163, (8, 197, 384), (75648, 384, 1), 0); del buf163  # reuse
        # Source Nodes: [cat_22, l__mod___blocks_2_norm_mlp, patch_embed_13], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf165, buf146, buf148, arg82_1, arg88_1, arg89_1, arg90_1, buf196, 1576, 384, grid=grid(1576), stream=stream0)
        del arg82_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf173 = buf119; del buf119  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (25088, 24), (24, 1), 0), reinterpret_tensor(arg97_1, (24, 48), (1, 24), 0), out=buf173)
        del arg97_1
        buf174 = reinterpret_tensor(buf148, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf148  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf173, buf174, 602112, grid=grid(602112), stream=stream0)
        buf175 = reinterpret_tensor(buf147, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf147  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf173, buf175, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf176 = reinterpret_tensor(buf126, (6272, 16, 16), (256, 16, 1), 0); del buf126  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf175, (6272, 6, 16), (96, 16, 1), 0), out=buf176)
        buf180 = reinterpret_tensor(buf122, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf122  # reuse
        # Source Nodes: [attn_18, attn_19], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf176, buf180, 100352, 16, grid=grid(100352), stream=stream0)
        buf179 = reinterpret_tensor(buf175, (25088, 24), (24, 1), 0); del buf175  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (25088, 24), (24, 1), 0), reinterpret_tensor(arg98_1, (24, 24), (1, 24), 0), out=buf179)
        del arg98_1
        buf181 = reinterpret_tensor(buf172, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf172  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf179, buf181, 602112, grid=grid(602112), stream=stream0)
        buf182 = reinterpret_tensor(buf179, (6272, 16, 6), (96, 6, 1), 0); del buf179  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf181, (6272, 16, 6), (96, 6, 1), 0), out=buf182)
        buf183 = reinterpret_tensor(buf181, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf181  # reuse
        # Source Nodes: [x_59], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf182, buf183, 602112, grid=grid(602112), stream=stream0)
        buf184 = reinterpret_tensor(buf182, (25088, 24), (24, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (25088, 24), (24, 1), 0), reinterpret_tensor(arg99_1, (24, 24), (1, 24), 0), out=buf184)
        del arg99_1
        buf185 = reinterpret_tensor(buf184, (1568, 16, 24), (384, 24, 1), 0); del buf184  # reuse
        buf189 = reinterpret_tensor(buf183, (1568, 16, 24), (384, 24, 1), 0); del buf183  # reuse
        # Source Nodes: [l__mod___blocks_3_norm_mlp_in, pixel_embed_10, pixel_embed_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf185, buf131, buf138, arg78_1, arg100_1, arg101_1, arg102_1, buf189, 25088, 24, grid=grid(25088), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg78_1
        buf190 = reinterpret_tensor(buf137, (25088, 96), (96, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (25088, 24), (24, 1), 0), reinterpret_tensor(arg103_1, (24, 96), (1, 24), 0), out=buf190)
        del arg103_1
        buf191 = reinterpret_tensor(buf190, (1568, 16, 96), (1536, 96, 1), 0); del buf190  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf191, arg104_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg104_1
        buf192 = reinterpret_tensor(buf189, (25088, 24), (24, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (25088, 96), (96, 1), 0), reinterpret_tensor(arg105_1, (96, 24), (1, 96), 0), out=buf192)
        del arg105_1
        buf200 = reinterpret_tensor(buf138, (1568, 16, 24), (384, 24, 1), 0); del buf138  # reuse
        buf225 = buf131; del buf131  # reuse
        # Source Nodes: [l__mod___blocks_3_norm1_proj, l__mod___blocks_4_norm_in, pixel_embed_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf185, buf192, arg106_1, arg107_1, arg108_1, arg123_1, arg124_1, buf200, buf225, 25088, 24, grid=grid(25088), stream=stream0)
        del arg107_1
        del arg108_1
        del arg123_1
        del arg124_1
        buf197 = reinterpret_tensor(buf144, (1576, 1536), (1536, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (1576, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 1536), (1, 384), 0), out=buf197)
        del arg91_1
        buf198 = reinterpret_tensor(buf197, (8, 197, 1536), (302592, 1536, 1), 0); del buf197  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf198, arg92_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg92_1
        buf199 = reinterpret_tensor(buf196, (1576, 384), (384, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg93_1, (1536, 384), (1, 1536), 0), out=buf199)
        del arg93_1
        buf201 = reinterpret_tensor(buf174, (1568, 384), (384, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1568, 384), (384, 1), 0), reinterpret_tensor(arg109_1, (384, 384), (1, 384), 0), out=buf201)
        del arg109_1
        buf202 = reinterpret_tensor(buf199, (8, 197, 384), (75648, 384, 1), 0); del buf199  # reuse
        buf206 = buf146; del buf146  # reuse
        # Source Nodes: [cat_21, l__mod___blocks_3_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf202, buf165, arg94_1, buf201, arg110_1, arg111_1, arg112_1, buf206, 1576, 384, grid=grid(1576), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        del arg94_1
        buf207 = buf153; del buf153  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf206, (1576, 384), (384, 1), 0), reinterpret_tensor(arg113_1, (384, 768), (1, 384), 0), out=buf207)
        del arg113_1
        buf208 = reinterpret_tensor(buf165, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf165  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf207, buf208, 605184, grid=grid(605184), stream=stream0)
        buf209 = reinterpret_tensor(buf154, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf154  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf207, buf209, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf210 = reinterpret_tensor(buf160, (48, 197, 197), (38809, 197, 1), 0); del buf160  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf209, (48, 64, 197), (12608, 197, 1), 0), out=buf210)
        buf214 = reinterpret_tensor(buf156, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf156  # reuse
        # Source Nodes: [attn_21, attn_22], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf210, buf214, 9456, 197, grid=grid(9456), stream=stream0)
        buf213 = reinterpret_tensor(buf209, (1576, 384), (384, 1), 0); del buf209  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf206, (1576, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 384), (1, 384), 0), out=buf213)
        del arg114_1
        buf215 = reinterpret_tensor(buf206, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf206  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf213, buf215, 605184, grid=grid(605184), stream=stream0)
        buf216 = reinterpret_tensor(buf213, (48, 197, 64), (12608, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf214, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf215, (48, 197, 64), (12608, 64, 1), 0), out=buf216)
        buf217 = reinterpret_tensor(buf215, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf216, buf217, 605184, grid=grid(605184), stream=stream0)
        buf218 = reinterpret_tensor(buf216, (1576, 384), (384, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (1576, 384), (384, 1), 0), reinterpret_tensor(arg115_1, (384, 384), (1, 384), 0), out=buf218)
        del arg115_1
        buf249 = reinterpret_tensor(buf217, (8, 197, 384), (75648, 384, 1), 0); del buf217  # reuse
        # Source Nodes: [l__mod___blocks_3_norm_mlp, patch_embed_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf202, buf218, arg116_1, arg117_1, arg118_1, buf249, 1576, 384, grid=grid(1576), stream=stream0)
        del arg117_1
        del arg118_1
        buf226 = buf173; del buf173  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (25088, 24), (24, 1), 0), reinterpret_tensor(arg125_1, (24, 48), (1, 24), 0), out=buf226)
        del arg125_1
        buf227 = reinterpret_tensor(buf201, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf201  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf226, buf227, 602112, grid=grid(602112), stream=stream0)
        buf228 = reinterpret_tensor(buf200, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf200  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf226, buf228, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf229 = reinterpret_tensor(buf180, (6272, 16, 16), (256, 16, 1), 0); del buf180  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf227, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf228, (6272, 6, 16), (96, 16, 1), 0), out=buf229)
        buf233 = reinterpret_tensor(buf176, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf176  # reuse
        # Source Nodes: [attn_24, attn_25], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf229, buf233, 100352, 16, grid=grid(100352), stream=stream0)
        buf232 = reinterpret_tensor(buf228, (25088, 24), (24, 1), 0); del buf228  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (25088, 24), (24, 1), 0), reinterpret_tensor(arg126_1, (24, 24), (1, 24), 0), out=buf232)
        del arg126_1
        buf234 = reinterpret_tensor(buf225, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf225  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf232, buf234, 602112, grid=grid(602112), stream=stream0)
        buf235 = reinterpret_tensor(buf232, (6272, 16, 6), (96, 6, 1), 0); del buf232  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf234, (6272, 16, 6), (96, 6, 1), 0), out=buf235)
        buf236 = reinterpret_tensor(buf234, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf234  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf235, buf236, 602112, grid=grid(602112), stream=stream0)
        buf237 = reinterpret_tensor(buf235, (25088, 24), (24, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (25088, 24), (24, 1), 0), reinterpret_tensor(arg127_1, (24, 24), (1, 24), 0), out=buf237)
        del arg127_1
        buf238 = reinterpret_tensor(buf237, (1568, 16, 24), (384, 24, 1), 0); del buf237  # reuse
        buf242 = reinterpret_tensor(buf236, (1568, 16, 24), (384, 24, 1), 0); del buf236  # reuse
        # Source Nodes: [l__mod___blocks_4_norm_mlp_in, pixel_embed_12, pixel_embed_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf238, buf185, buf192, arg106_1, arg128_1, arg129_1, arg130_1, buf242, 25088, 24, grid=grid(25088), stream=stream0)
        del arg106_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf243 = reinterpret_tensor(buf191, (25088, 96), (96, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (25088, 24), (24, 1), 0), reinterpret_tensor(arg131_1, (24, 96), (1, 24), 0), out=buf243)
        del arg131_1
        buf244 = reinterpret_tensor(buf243, (1568, 16, 96), (1536, 96, 1), 0); del buf243  # reuse
        # Source Nodes: [x_81], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf244, arg132_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg132_1
        buf245 = reinterpret_tensor(buf242, (25088, 24), (24, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (25088, 96), (96, 1), 0), reinterpret_tensor(arg133_1, (96, 24), (1, 96), 0), out=buf245)
        del arg133_1
        buf254 = reinterpret_tensor(buf192, (1568, 16, 24), (384, 24, 1), 0); del buf192  # reuse
        buf279 = buf185; del buf185  # reuse
        # Source Nodes: [l__mod___blocks_4_norm1_proj, l__mod___blocks_5_norm_in, pixel_embed_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf238, buf245, arg134_1, arg135_1, arg136_1, arg151_1, arg152_1, buf254, buf279, 25088, 24, grid=grid(25088), stream=stream0)
        del arg135_1
        del arg136_1
        del arg151_1
        del arg152_1
        buf250 = reinterpret_tensor(buf198, (1576, 1536), (1536, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (1576, 384), (384, 1), 0), reinterpret_tensor(arg119_1, (384, 1536), (1, 384), 0), out=buf250)
        del arg119_1
        buf251 = reinterpret_tensor(buf250, (8, 197, 1536), (302592, 1536, 1), 0); del buf250  # reuse
        # Source Nodes: [x_72], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf251, arg120_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg120_1
        buf252 = reinterpret_tensor(buf249, (1576, 384), (384, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg121_1, (1536, 384), (1, 1536), 0), out=buf252)
        del arg121_1
        buf253 = reinterpret_tensor(buf252, (8, 197, 384), (75648, 384, 1), 0); del buf252  # reuse
        # Source Nodes: [patch_embed_17, patch_embed_19], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf253, buf202, buf218, arg116_1, arg122_1, 605184, grid=grid(605184), stream=stream0)
        del arg116_1
        del arg122_1
        buf255 = reinterpret_tensor(buf227, (1568, 384), (384, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf254, (1568, 384), (384, 1), 0), reinterpret_tensor(arg137_1, (384, 384), (1, 384), 0), out=buf255)
        del arg137_1
        buf259 = reinterpret_tensor(buf218, (8, 197, 384), (75648, 384, 1), 0); del buf218  # reuse
        # Source Nodes: [cat_20, l__mod___blocks_4_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf253, buf255, arg138_1, arg139_1, arg140_1, buf259, 1576, 384, grid=grid(1576), stream=stream0)
        del arg139_1
        del arg140_1
        buf260 = buf207; del buf207  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (1576, 384), (384, 1), 0), reinterpret_tensor(arg141_1, (384, 768), (1, 384), 0), out=buf260)
        del arg141_1
        buf261 = reinterpret_tensor(buf202, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf202  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf260, buf261, 605184, grid=grid(605184), stream=stream0)
        buf262 = reinterpret_tensor(buf208, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf208  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf260, buf262, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf263 = reinterpret_tensor(buf214, (48, 197, 197), (38809, 197, 1), 0); del buf214  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf262, (48, 64, 197), (12608, 197, 1), 0), out=buf263)
        buf267 = reinterpret_tensor(buf210, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf210  # reuse
        # Source Nodes: [attn_27, attn_28], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf263, buf267, 9456, 197, grid=grid(9456), stream=stream0)
        buf266 = reinterpret_tensor(buf262, (1576, 384), (384, 1), 0); del buf262  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (1576, 384), (384, 1), 0), reinterpret_tensor(arg142_1, (384, 384), (1, 384), 0), out=buf266)
        del arg142_1
        buf268 = reinterpret_tensor(buf259, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf266, buf268, 605184, grid=grid(605184), stream=stream0)
        buf269 = reinterpret_tensor(buf266, (48, 197, 64), (12608, 64, 1), 0); del buf266  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf268, (48, 197, 64), (12608, 64, 1), 0), out=buf269)
        buf270 = reinterpret_tensor(buf268, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf268  # reuse
        # Source Nodes: [x_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf269, buf270, 605184, grid=grid(605184), stream=stream0)
        buf271 = reinterpret_tensor(buf269, (1576, 384), (384, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf270, (1576, 384), (384, 1), 0), reinterpret_tensor(arg143_1, (384, 384), (1, 384), 0), out=buf271)
        del arg143_1
        buf272 = reinterpret_tensor(buf271, (8, 197, 384), (75648, 384, 1), 0); del buf271  # reuse
        buf303 = reinterpret_tensor(buf270, (8, 197, 384), (75648, 384, 1), 0); del buf270  # reuse
        # Source Nodes: [cat_20, l__mod___blocks_4_norm_mlp, patch_embed_21], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf272, buf253, buf255, arg138_1, arg144_1, arg145_1, arg146_1, buf303, 1576, 384, grid=grid(1576), stream=stream0)
        del arg138_1
        del arg144_1
        del arg145_1
        del arg146_1
        buf280 = buf226; del buf226  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (25088, 24), (24, 1), 0), reinterpret_tensor(arg153_1, (24, 48), (1, 24), 0), out=buf280)
        del arg153_1
        buf281 = reinterpret_tensor(buf255, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf255  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf280, buf281, 602112, grid=grid(602112), stream=stream0)
        buf282 = reinterpret_tensor(buf254, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf254  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf280, buf282, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf283 = reinterpret_tensor(buf233, (6272, 16, 16), (256, 16, 1), 0); del buf233  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf281, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf282, (6272, 6, 16), (96, 16, 1), 0), out=buf283)
        buf287 = reinterpret_tensor(buf229, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf229  # reuse
        # Source Nodes: [attn_30, attn_31], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf283, buf287, 100352, 16, grid=grid(100352), stream=stream0)
        buf286 = reinterpret_tensor(buf282, (25088, 24), (24, 1), 0); del buf282  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (25088, 24), (24, 1), 0), reinterpret_tensor(arg154_1, (24, 24), (1, 24), 0), out=buf286)
        del arg154_1
        buf288 = reinterpret_tensor(buf279, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf279  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf286, buf288, 602112, grid=grid(602112), stream=stream0)
        buf289 = reinterpret_tensor(buf286, (6272, 16, 6), (96, 6, 1), 0); del buf286  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf287, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf288, (6272, 16, 6), (96, 6, 1), 0), out=buf289)
        buf290 = reinterpret_tensor(buf288, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf288  # reuse
        # Source Nodes: [x_95], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf289, buf290, 602112, grid=grid(602112), stream=stream0)
        buf291 = reinterpret_tensor(buf289, (25088, 24), (24, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (25088, 24), (24, 1), 0), reinterpret_tensor(arg155_1, (24, 24), (1, 24), 0), out=buf291)
        del arg155_1
        buf292 = reinterpret_tensor(buf291, (1568, 16, 24), (384, 24, 1), 0); del buf291  # reuse
        buf296 = reinterpret_tensor(buf290, (1568, 16, 24), (384, 24, 1), 0); del buf290  # reuse
        # Source Nodes: [l__mod___blocks_5_norm_mlp_in, pixel_embed_15, pixel_embed_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf292, buf238, buf245, arg134_1, arg156_1, arg157_1, arg158_1, buf296, 25088, 24, grid=grid(25088), stream=stream0)
        del arg134_1
        del arg156_1
        del arg157_1
        del arg158_1
        buf297 = reinterpret_tensor(buf244, (25088, 96), (96, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (25088, 24), (24, 1), 0), reinterpret_tensor(arg159_1, (24, 96), (1, 24), 0), out=buf297)
        del arg159_1
        buf298 = reinterpret_tensor(buf297, (1568, 16, 96), (1536, 96, 1), 0); del buf297  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf298, arg160_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg160_1
        buf299 = reinterpret_tensor(buf296, (25088, 24), (24, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (25088, 96), (96, 1), 0), reinterpret_tensor(arg161_1, (96, 24), (1, 96), 0), out=buf299)
        del arg161_1
        buf307 = reinterpret_tensor(buf245, (1568, 16, 24), (384, 24, 1), 0); del buf245  # reuse
        buf332 = buf238; del buf238  # reuse
        # Source Nodes: [l__mod___blocks_5_norm1_proj, l__mod___blocks_6_norm_in, pixel_embed_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf292, buf299, arg162_1, arg163_1, arg164_1, arg179_1, arg180_1, buf307, buf332, 25088, 24, grid=grid(25088), stream=stream0)
        del arg163_1
        del arg164_1
        del arg179_1
        del arg180_1
        buf304 = reinterpret_tensor(buf251, (1576, 1536), (1536, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (1576, 384), (384, 1), 0), reinterpret_tensor(arg147_1, (384, 1536), (1, 384), 0), out=buf304)
        del arg147_1
        buf305 = reinterpret_tensor(buf304, (8, 197, 1536), (302592, 1536, 1), 0); del buf304  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf305, arg148_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg148_1
        buf306 = reinterpret_tensor(buf303, (1576, 384), (384, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf305, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg149_1, (1536, 384), (1, 1536), 0), out=buf306)
        del arg149_1
        buf308 = reinterpret_tensor(buf281, (1568, 384), (384, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (1568, 384), (384, 1), 0), reinterpret_tensor(arg165_1, (384, 384), (1, 384), 0), out=buf308)
        del arg165_1
        buf309 = reinterpret_tensor(buf306, (8, 197, 384), (75648, 384, 1), 0); del buf306  # reuse
        buf313 = buf253; del buf253  # reuse
        # Source Nodes: [cat_19, l__mod___blocks_5_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf309, buf272, arg150_1, buf308, arg166_1, arg167_1, arg168_1, buf313, 1576, 384, grid=grid(1576), stream=stream0)
        del arg150_1
        del arg166_1
        del arg167_1
        del arg168_1
        buf314 = buf260; del buf260  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (1576, 384), (384, 1), 0), reinterpret_tensor(arg169_1, (384, 768), (1, 384), 0), out=buf314)
        del arg169_1
        buf315 = reinterpret_tensor(buf272, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf272  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf314, buf315, 605184, grid=grid(605184), stream=stream0)
        buf316 = reinterpret_tensor(buf261, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf261  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf314, buf316, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf317 = reinterpret_tensor(buf267, (48, 197, 197), (38809, 197, 1), 0); del buf267  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf316, (48, 64, 197), (12608, 197, 1), 0), out=buf317)
        buf321 = reinterpret_tensor(buf263, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf263  # reuse
        # Source Nodes: [attn_33, attn_34], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf317, buf321, 9456, 197, grid=grid(9456), stream=stream0)
        buf320 = reinterpret_tensor(buf316, (1576, 384), (384, 1), 0); del buf316  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (1576, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 384), (1, 384), 0), out=buf320)
        del arg170_1
        buf322 = reinterpret_tensor(buf313, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf313  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf320, buf322, 605184, grid=grid(605184), stream=stream0)
        buf323 = reinterpret_tensor(buf320, (48, 197, 64), (12608, 64, 1), 0); del buf320  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf322, (48, 197, 64), (12608, 64, 1), 0), out=buf323)
        buf324 = reinterpret_tensor(buf322, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf322  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf323, buf324, 605184, grid=grid(605184), stream=stream0)
        buf325 = reinterpret_tensor(buf323, (1576, 384), (384, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (1576, 384), (384, 1), 0), reinterpret_tensor(arg171_1, (384, 384), (1, 384), 0), out=buf325)
        del arg171_1
        buf356 = reinterpret_tensor(buf324, (8, 197, 384), (75648, 384, 1), 0); del buf324  # reuse
        # Source Nodes: [l__mod___blocks_5_norm_mlp, patch_embed_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf309, buf325, arg172_1, arg173_1, arg174_1, buf356, 1576, 384, grid=grid(1576), stream=stream0)
        del arg173_1
        del arg174_1
        buf333 = buf280; del buf280  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (25088, 24), (24, 1), 0), reinterpret_tensor(arg181_1, (24, 48), (1, 24), 0), out=buf333)
        del arg181_1
        buf334 = reinterpret_tensor(buf308, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf308  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf333, buf334, 602112, grid=grid(602112), stream=stream0)
        buf335 = reinterpret_tensor(buf307, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf307  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf333, buf335, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf336 = reinterpret_tensor(buf287, (6272, 16, 16), (256, 16, 1), 0); del buf287  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf334, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf335, (6272, 6, 16), (96, 16, 1), 0), out=buf336)
        buf340 = reinterpret_tensor(buf283, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf283  # reuse
        # Source Nodes: [attn_36, attn_37], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf336, buf340, 100352, 16, grid=grid(100352), stream=stream0)
        buf339 = reinterpret_tensor(buf335, (25088, 24), (24, 1), 0); del buf335  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (25088, 24), (24, 1), 0), reinterpret_tensor(arg182_1, (24, 24), (1, 24), 0), out=buf339)
        del arg182_1
        buf341 = reinterpret_tensor(buf332, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf332  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf339, buf341, 602112, grid=grid(602112), stream=stream0)
        buf342 = reinterpret_tensor(buf339, (6272, 16, 6), (96, 6, 1), 0); del buf339  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf341, (6272, 16, 6), (96, 6, 1), 0), out=buf342)
        buf343 = reinterpret_tensor(buf341, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf341  # reuse
        # Source Nodes: [x_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf342, buf343, 602112, grid=grid(602112), stream=stream0)
        buf344 = reinterpret_tensor(buf342, (25088, 24), (24, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (25088, 24), (24, 1), 0), reinterpret_tensor(arg183_1, (24, 24), (1, 24), 0), out=buf344)
        del arg183_1
        buf345 = reinterpret_tensor(buf344, (1568, 16, 24), (384, 24, 1), 0); del buf344  # reuse
        buf349 = reinterpret_tensor(buf343, (1568, 16, 24), (384, 24, 1), 0); del buf343  # reuse
        # Source Nodes: [l__mod___blocks_6_norm_mlp_in, pixel_embed_18, pixel_embed_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf345, buf292, buf299, arg162_1, arg184_1, arg185_1, arg186_1, buf349, 25088, 24, grid=grid(25088), stream=stream0)
        del arg162_1
        del arg184_1
        del arg185_1
        del arg186_1
        buf350 = reinterpret_tensor(buf298, (25088, 96), (96, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf349, (25088, 24), (24, 1), 0), reinterpret_tensor(arg187_1, (24, 96), (1, 24), 0), out=buf350)
        del arg187_1
        buf351 = reinterpret_tensor(buf350, (1568, 16, 96), (1536, 96, 1), 0); del buf350  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf351, arg188_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg188_1
        buf352 = reinterpret_tensor(buf349, (25088, 24), (24, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (25088, 96), (96, 1), 0), reinterpret_tensor(arg189_1, (96, 24), (1, 96), 0), out=buf352)
        del arg189_1
        buf361 = reinterpret_tensor(buf299, (1568, 16, 24), (384, 24, 1), 0); del buf299  # reuse
        buf386 = buf292; del buf292  # reuse
        # Source Nodes: [l__mod___blocks_6_norm1_proj, l__mod___blocks_7_norm_in, pixel_embed_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf345, buf352, arg190_1, arg191_1, arg192_1, arg207_1, arg208_1, buf361, buf386, 25088, 24, grid=grid(25088), stream=stream0)
        del arg191_1
        del arg192_1
        del arg207_1
        del arg208_1
        buf357 = reinterpret_tensor(buf305, (1576, 1536), (1536, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (1576, 384), (384, 1), 0), reinterpret_tensor(arg175_1, (384, 1536), (1, 384), 0), out=buf357)
        del arg175_1
        buf358 = reinterpret_tensor(buf357, (8, 197, 1536), (302592, 1536, 1), 0); del buf357  # reuse
        # Source Nodes: [x_108], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf358, arg176_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg176_1
        buf359 = reinterpret_tensor(buf356, (1576, 384), (384, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg177_1, (1536, 384), (1, 1536), 0), out=buf359)
        del arg177_1
        buf360 = reinterpret_tensor(buf359, (8, 197, 384), (75648, 384, 1), 0); del buf359  # reuse
        # Source Nodes: [patch_embed_25, patch_embed_27], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf360, buf309, buf325, arg172_1, arg178_1, 605184, grid=grid(605184), stream=stream0)
        del arg172_1
        del arg178_1
        buf362 = reinterpret_tensor(buf334, (1568, 384), (384, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf361, (1568, 384), (384, 1), 0), reinterpret_tensor(arg193_1, (384, 384), (1, 384), 0), out=buf362)
        del arg193_1
        buf366 = reinterpret_tensor(buf325, (8, 197, 384), (75648, 384, 1), 0); del buf325  # reuse
        # Source Nodes: [cat_18, l__mod___blocks_6_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf360, buf362, arg194_1, arg195_1, arg196_1, buf366, 1576, 384, grid=grid(1576), stream=stream0)
        del arg195_1
        del arg196_1
        buf367 = buf314; del buf314  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (1576, 384), (384, 1), 0), reinterpret_tensor(arg197_1, (384, 768), (1, 384), 0), out=buf367)
        del arg197_1
        buf368 = reinterpret_tensor(buf309, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf309  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf367, buf368, 605184, grid=grid(605184), stream=stream0)
        buf369 = reinterpret_tensor(buf315, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf315  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf367, buf369, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf370 = reinterpret_tensor(buf321, (48, 197, 197), (38809, 197, 1), 0); del buf321  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf368, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf369, (48, 64, 197), (12608, 197, 1), 0), out=buf370)
        buf374 = reinterpret_tensor(buf317, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf317  # reuse
        # Source Nodes: [attn_39, attn_40], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf370, buf374, 9456, 197, grid=grid(9456), stream=stream0)
        buf373 = reinterpret_tensor(buf369, (1576, 384), (384, 1), 0); del buf369  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (1576, 384), (384, 1), 0), reinterpret_tensor(arg198_1, (384, 384), (1, 384), 0), out=buf373)
        del arg198_1
        buf375 = reinterpret_tensor(buf366, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf366  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf373, buf375, 605184, grid=grid(605184), stream=stream0)
        buf376 = reinterpret_tensor(buf373, (48, 197, 64), (12608, 64, 1), 0); del buf373  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf374, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf375, (48, 197, 64), (12608, 64, 1), 0), out=buf376)
        buf377 = reinterpret_tensor(buf375, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf375  # reuse
        # Source Nodes: [x_122], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf376, buf377, 605184, grid=grid(605184), stream=stream0)
        buf378 = reinterpret_tensor(buf376, (1576, 384), (384, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf377, (1576, 384), (384, 1), 0), reinterpret_tensor(arg199_1, (384, 384), (1, 384), 0), out=buf378)
        del arg199_1
        buf379 = reinterpret_tensor(buf378, (8, 197, 384), (75648, 384, 1), 0); del buf378  # reuse
        buf410 = reinterpret_tensor(buf377, (8, 197, 384), (75648, 384, 1), 0); del buf377  # reuse
        # Source Nodes: [cat_18, l__mod___blocks_6_norm_mlp, patch_embed_29], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf379, buf360, buf362, arg194_1, arg200_1, arg201_1, arg202_1, buf410, 1576, 384, grid=grid(1576), stream=stream0)
        del arg194_1
        del arg200_1
        del arg201_1
        del arg202_1
        buf387 = buf333; del buf333  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (25088, 24), (24, 1), 0), reinterpret_tensor(arg209_1, (24, 48), (1, 24), 0), out=buf387)
        del arg209_1
        buf388 = reinterpret_tensor(buf362, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf362  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf387, buf388, 602112, grid=grid(602112), stream=stream0)
        buf389 = reinterpret_tensor(buf361, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf361  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf387, buf389, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf390 = reinterpret_tensor(buf340, (6272, 16, 16), (256, 16, 1), 0); del buf340  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf388, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf389, (6272, 6, 16), (96, 16, 1), 0), out=buf390)
        buf394 = reinterpret_tensor(buf336, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf336  # reuse
        # Source Nodes: [attn_42, attn_43], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf390, buf394, 100352, 16, grid=grid(100352), stream=stream0)
        buf393 = reinterpret_tensor(buf389, (25088, 24), (24, 1), 0); del buf389  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (25088, 24), (24, 1), 0), reinterpret_tensor(arg210_1, (24, 24), (1, 24), 0), out=buf393)
        del arg210_1
        buf395 = reinterpret_tensor(buf386, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf386  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf393, buf395, 602112, grid=grid(602112), stream=stream0)
        buf396 = reinterpret_tensor(buf393, (6272, 16, 6), (96, 6, 1), 0); del buf393  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf394, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf395, (6272, 16, 6), (96, 6, 1), 0), out=buf396)
        buf397 = reinterpret_tensor(buf395, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf395  # reuse
        # Source Nodes: [x_131], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf396, buf397, 602112, grid=grid(602112), stream=stream0)
        buf398 = reinterpret_tensor(buf396, (25088, 24), (24, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (25088, 24), (24, 1), 0), reinterpret_tensor(arg211_1, (24, 24), (1, 24), 0), out=buf398)
        del arg211_1
        buf399 = reinterpret_tensor(buf398, (1568, 16, 24), (384, 24, 1), 0); del buf398  # reuse
        buf403 = reinterpret_tensor(buf397, (1568, 16, 24), (384, 24, 1), 0); del buf397  # reuse
        # Source Nodes: [l__mod___blocks_7_norm_mlp_in, pixel_embed_21, pixel_embed_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf399, buf345, buf352, arg190_1, arg212_1, arg213_1, arg214_1, buf403, 25088, 24, grid=grid(25088), stream=stream0)
        del arg190_1
        del arg212_1
        del arg213_1
        del arg214_1
        buf404 = reinterpret_tensor(buf351, (25088, 96), (96, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (25088, 24), (24, 1), 0), reinterpret_tensor(arg215_1, (24, 96), (1, 24), 0), out=buf404)
        del arg215_1
        buf405 = reinterpret_tensor(buf404, (1568, 16, 96), (1536, 96, 1), 0); del buf404  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf405, arg216_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg216_1
        buf406 = reinterpret_tensor(buf403, (25088, 24), (24, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf405, (25088, 96), (96, 1), 0), reinterpret_tensor(arg217_1, (96, 24), (1, 96), 0), out=buf406)
        del arg217_1
        buf414 = reinterpret_tensor(buf352, (1568, 16, 24), (384, 24, 1), 0); del buf352  # reuse
        buf439 = buf345; del buf345  # reuse
        # Source Nodes: [l__mod___blocks_7_norm1_proj, l__mod___blocks_8_norm_in, pixel_embed_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf399, buf406, arg218_1, arg219_1, arg220_1, arg235_1, arg236_1, buf414, buf439, 25088, 24, grid=grid(25088), stream=stream0)
        del arg219_1
        del arg220_1
        del arg235_1
        del arg236_1
        buf411 = reinterpret_tensor(buf358, (1576, 1536), (1536, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (1576, 384), (384, 1), 0), reinterpret_tensor(arg203_1, (384, 1536), (1, 384), 0), out=buf411)
        del arg203_1
        buf412 = reinterpret_tensor(buf411, (8, 197, 1536), (302592, 1536, 1), 0); del buf411  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf412, arg204_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg204_1
        buf413 = reinterpret_tensor(buf410, (1576, 384), (384, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg205_1, (1536, 384), (1, 1536), 0), out=buf413)
        del arg205_1
        buf415 = reinterpret_tensor(buf388, (1568, 384), (384, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf414, (1568, 384), (384, 1), 0), reinterpret_tensor(arg221_1, (384, 384), (1, 384), 0), out=buf415)
        del arg221_1
        buf416 = reinterpret_tensor(buf413, (8, 197, 384), (75648, 384, 1), 0); del buf413  # reuse
        buf420 = buf360; del buf360  # reuse
        # Source Nodes: [cat_17, l__mod___blocks_7_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf416, buf379, arg206_1, buf415, arg222_1, arg223_1, arg224_1, buf420, 1576, 384, grid=grid(1576), stream=stream0)
        del arg206_1
        del arg222_1
        del arg223_1
        del arg224_1
        buf421 = buf367; del buf367  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (1576, 384), (384, 1), 0), reinterpret_tensor(arg225_1, (384, 768), (1, 384), 0), out=buf421)
        del arg225_1
        buf422 = reinterpret_tensor(buf379, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf379  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf421, buf422, 605184, grid=grid(605184), stream=stream0)
        buf423 = reinterpret_tensor(buf368, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf368  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf421, buf423, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf424 = reinterpret_tensor(buf374, (48, 197, 197), (38809, 197, 1), 0); del buf374  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf422, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf423, (48, 64, 197), (12608, 197, 1), 0), out=buf424)
        buf428 = reinterpret_tensor(buf370, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf370  # reuse
        # Source Nodes: [attn_45, attn_46], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf424, buf428, 9456, 197, grid=grid(9456), stream=stream0)
        buf427 = reinterpret_tensor(buf423, (1576, 384), (384, 1), 0); del buf423  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (1576, 384), (384, 1), 0), reinterpret_tensor(arg226_1, (384, 384), (1, 384), 0), out=buf427)
        del arg226_1
        buf429 = reinterpret_tensor(buf420, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf420  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf427, buf429, 605184, grid=grid(605184), stream=stream0)
        buf430 = reinterpret_tensor(buf427, (48, 197, 64), (12608, 64, 1), 0); del buf427  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf428, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf429, (48, 197, 64), (12608, 64, 1), 0), out=buf430)
        buf431 = reinterpret_tensor(buf429, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf429  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf430, buf431, 605184, grid=grid(605184), stream=stream0)
        buf432 = reinterpret_tensor(buf430, (1576, 384), (384, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf431, (1576, 384), (384, 1), 0), reinterpret_tensor(arg227_1, (384, 384), (1, 384), 0), out=buf432)
        del arg227_1
        buf463 = reinterpret_tensor(buf431, (8, 197, 384), (75648, 384, 1), 0); del buf431  # reuse
        # Source Nodes: [l__mod___blocks_7_norm_mlp, patch_embed_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf416, buf432, arg228_1, arg229_1, arg230_1, buf463, 1576, 384, grid=grid(1576), stream=stream0)
        del arg229_1
        del arg230_1
        buf440 = buf387; del buf387  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (25088, 24), (24, 1), 0), reinterpret_tensor(arg237_1, (24, 48), (1, 24), 0), out=buf440)
        del arg237_1
        buf441 = reinterpret_tensor(buf415, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf415  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf440, buf441, 602112, grid=grid(602112), stream=stream0)
        buf442 = reinterpret_tensor(buf414, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf414  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf440, buf442, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf443 = reinterpret_tensor(buf394, (6272, 16, 16), (256, 16, 1), 0); del buf394  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf442, (6272, 6, 16), (96, 16, 1), 0), out=buf443)
        buf447 = reinterpret_tensor(buf390, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf390  # reuse
        # Source Nodes: [attn_48, attn_49], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf443, buf447, 100352, 16, grid=grid(100352), stream=stream0)
        buf446 = reinterpret_tensor(buf442, (25088, 24), (24, 1), 0); del buf442  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (25088, 24), (24, 1), 0), reinterpret_tensor(arg238_1, (24, 24), (1, 24), 0), out=buf446)
        del arg238_1
        buf448 = reinterpret_tensor(buf439, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf439  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf446, buf448, 602112, grid=grid(602112), stream=stream0)
        buf449 = reinterpret_tensor(buf446, (6272, 16, 6), (96, 6, 1), 0); del buf446  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf448, (6272, 16, 6), (96, 6, 1), 0), out=buf449)
        buf450 = reinterpret_tensor(buf448, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf448  # reuse
        # Source Nodes: [x_149], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf449, buf450, 602112, grid=grid(602112), stream=stream0)
        buf451 = reinterpret_tensor(buf449, (25088, 24), (24, 1), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (25088, 24), (24, 1), 0), reinterpret_tensor(arg239_1, (24, 24), (1, 24), 0), out=buf451)
        del arg239_1
        buf452 = reinterpret_tensor(buf451, (1568, 16, 24), (384, 24, 1), 0); del buf451  # reuse
        buf456 = reinterpret_tensor(buf450, (1568, 16, 24), (384, 24, 1), 0); del buf450  # reuse
        # Source Nodes: [l__mod___blocks_8_norm_mlp_in, pixel_embed_24, pixel_embed_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf452, buf399, buf406, arg218_1, arg240_1, arg241_1, arg242_1, buf456, 25088, 24, grid=grid(25088), stream=stream0)
        del arg218_1
        del arg240_1
        del arg241_1
        del arg242_1
        buf457 = reinterpret_tensor(buf405, (25088, 96), (96, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf456, (25088, 24), (24, 1), 0), reinterpret_tensor(arg243_1, (24, 96), (1, 24), 0), out=buf457)
        del arg243_1
        buf458 = reinterpret_tensor(buf457, (1568, 16, 96), (1536, 96, 1), 0); del buf457  # reuse
        # Source Nodes: [x_153], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf458, arg244_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg244_1
        buf459 = reinterpret_tensor(buf456, (25088, 24), (24, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (25088, 96), (96, 1), 0), reinterpret_tensor(arg245_1, (96, 24), (1, 96), 0), out=buf459)
        del arg245_1
        buf468 = reinterpret_tensor(buf406, (1568, 16, 24), (384, 24, 1), 0); del buf406  # reuse
        buf493 = buf399; del buf399  # reuse
        # Source Nodes: [l__mod___blocks_8_norm1_proj, l__mod___blocks_9_norm_in, pixel_embed_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf452, buf459, arg246_1, arg247_1, arg248_1, arg263_1, arg264_1, buf468, buf493, 25088, 24, grid=grid(25088), stream=stream0)
        del arg247_1
        del arg248_1
        del arg263_1
        del arg264_1
        buf464 = reinterpret_tensor(buf412, (1576, 1536), (1536, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (1576, 384), (384, 1), 0), reinterpret_tensor(arg231_1, (384, 1536), (1, 384), 0), out=buf464)
        del arg231_1
        buf465 = reinterpret_tensor(buf464, (8, 197, 1536), (302592, 1536, 1), 0); del buf464  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf465, arg232_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg232_1
        buf466 = reinterpret_tensor(buf463, (1576, 384), (384, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg233_1, (1536, 384), (1, 1536), 0), out=buf466)
        del arg233_1
        buf467 = reinterpret_tensor(buf466, (8, 197, 384), (75648, 384, 1), 0); del buf466  # reuse
        # Source Nodes: [patch_embed_33, patch_embed_35], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf467, buf416, buf432, arg228_1, arg234_1, 605184, grid=grid(605184), stream=stream0)
        del arg228_1
        del arg234_1
        buf469 = reinterpret_tensor(buf441, (1568, 384), (384, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf468, (1568, 384), (384, 1), 0), reinterpret_tensor(arg249_1, (384, 384), (1, 384), 0), out=buf469)
        del arg249_1
        buf473 = reinterpret_tensor(buf432, (8, 197, 384), (75648, 384, 1), 0); del buf432  # reuse
        # Source Nodes: [cat_16, l__mod___blocks_8_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf467, buf469, arg250_1, arg251_1, arg252_1, buf473, 1576, 384, grid=grid(1576), stream=stream0)
        del arg251_1
        del arg252_1
        buf474 = buf421; del buf421  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1576, 384), (384, 1), 0), reinterpret_tensor(arg253_1, (384, 768), (1, 384), 0), out=buf474)
        del arg253_1
        buf475 = reinterpret_tensor(buf416, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf416  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf474, buf475, 605184, grid=grid(605184), stream=stream0)
        buf476 = reinterpret_tensor(buf422, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf422  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf474, buf476, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf477 = reinterpret_tensor(buf428, (48, 197, 197), (38809, 197, 1), 0); del buf428  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf475, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf476, (48, 64, 197), (12608, 197, 1), 0), out=buf477)
        buf481 = reinterpret_tensor(buf424, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf424  # reuse
        # Source Nodes: [attn_51, attn_52], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf477, buf481, 9456, 197, grid=grid(9456), stream=stream0)
        buf480 = reinterpret_tensor(buf476, (1576, 384), (384, 1), 0); del buf476  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1576, 384), (384, 1), 0), reinterpret_tensor(arg254_1, (384, 384), (1, 384), 0), out=buf480)
        del arg254_1
        buf482 = reinterpret_tensor(buf473, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf473  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf480, buf482, 605184, grid=grid(605184), stream=stream0)
        buf483 = reinterpret_tensor(buf480, (48, 197, 64), (12608, 64, 1), 0); del buf480  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf481, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf482, (48, 197, 64), (12608, 64, 1), 0), out=buf483)
        buf484 = reinterpret_tensor(buf482, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf482  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf483, buf484, 605184, grid=grid(605184), stream=stream0)
        buf485 = reinterpret_tensor(buf483, (1576, 384), (384, 1), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (1576, 384), (384, 1), 0), reinterpret_tensor(arg255_1, (384, 384), (1, 384), 0), out=buf485)
        del arg255_1
        buf486 = reinterpret_tensor(buf485, (8, 197, 384), (75648, 384, 1), 0); del buf485  # reuse
        buf517 = reinterpret_tensor(buf484, (8, 197, 384), (75648, 384, 1), 0); del buf484  # reuse
        # Source Nodes: [cat_16, l__mod___blocks_8_norm_mlp, patch_embed_37], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf486, buf467, buf469, arg250_1, arg256_1, arg257_1, arg258_1, buf517, 1576, 384, grid=grid(1576), stream=stream0)
        del arg250_1
        del arg256_1
        del arg257_1
        del arg258_1
        buf494 = buf440; del buf440  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (25088, 24), (24, 1), 0), reinterpret_tensor(arg265_1, (24, 48), (1, 24), 0), out=buf494)
        del arg265_1
        buf495 = reinterpret_tensor(buf469, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf469  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf494, buf495, 602112, grid=grid(602112), stream=stream0)
        buf496 = reinterpret_tensor(buf468, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf468  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf494, buf496, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf497 = reinterpret_tensor(buf447, (6272, 16, 16), (256, 16, 1), 0); del buf447  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf496, (6272, 6, 16), (96, 16, 1), 0), out=buf497)
        buf501 = reinterpret_tensor(buf443, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf443  # reuse
        # Source Nodes: [attn_54, attn_55], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf497, buf501, 100352, 16, grid=grid(100352), stream=stream0)
        buf500 = reinterpret_tensor(buf496, (25088, 24), (24, 1), 0); del buf496  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (25088, 24), (24, 1), 0), reinterpret_tensor(arg266_1, (24, 24), (1, 24), 0), out=buf500)
        del arg266_1
        buf502 = reinterpret_tensor(buf493, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf493  # reuse
        # Source Nodes: [matmul_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf500, buf502, 602112, grid=grid(602112), stream=stream0)
        buf503 = reinterpret_tensor(buf500, (6272, 16, 6), (96, 6, 1), 0); del buf500  # reuse
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf501, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf502, (6272, 16, 6), (96, 6, 1), 0), out=buf503)
        buf504 = reinterpret_tensor(buf502, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf502  # reuse
        # Source Nodes: [x_167], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf503, buf504, 602112, grid=grid(602112), stream=stream0)
        buf505 = reinterpret_tensor(buf503, (25088, 24), (24, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf504, (25088, 24), (24, 1), 0), reinterpret_tensor(arg267_1, (24, 24), (1, 24), 0), out=buf505)
        del arg267_1
        buf506 = reinterpret_tensor(buf505, (1568, 16, 24), (384, 24, 1), 0); del buf505  # reuse
        buf510 = reinterpret_tensor(buf504, (1568, 16, 24), (384, 24, 1), 0); del buf504  # reuse
        # Source Nodes: [l__mod___blocks_9_norm_mlp_in, pixel_embed_27, pixel_embed_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf506, buf452, buf459, arg246_1, arg268_1, arg269_1, arg270_1, buf510, 25088, 24, grid=grid(25088), stream=stream0)
        del arg246_1
        del arg268_1
        del arg269_1
        del arg270_1
        buf511 = reinterpret_tensor(buf458, (25088, 96), (96, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf510, (25088, 24), (24, 1), 0), reinterpret_tensor(arg271_1, (24, 96), (1, 24), 0), out=buf511)
        del arg271_1
        buf512 = reinterpret_tensor(buf511, (1568, 16, 96), (1536, 96, 1), 0); del buf511  # reuse
        # Source Nodes: [x_171], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf512, arg272_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg272_1
        buf513 = reinterpret_tensor(buf510, (25088, 24), (24, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf512, (25088, 96), (96, 1), 0), reinterpret_tensor(arg273_1, (96, 24), (1, 96), 0), out=buf513)
        del arg273_1
        buf521 = reinterpret_tensor(buf459, (1568, 16, 24), (384, 24, 1), 0); del buf459  # reuse
        buf546 = buf452; del buf452  # reuse
        # Source Nodes: [l__mod___blocks_10_norm_in, l__mod___blocks_9_norm1_proj, pixel_embed_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf506, buf513, arg274_1, arg275_1, arg276_1, arg291_1, arg292_1, buf521, buf546, 25088, 24, grid=grid(25088), stream=stream0)
        del arg275_1
        del arg276_1
        del arg291_1
        del arg292_1
        buf518 = reinterpret_tensor(buf465, (1576, 1536), (1536, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf517, (1576, 384), (384, 1), 0), reinterpret_tensor(arg259_1, (384, 1536), (1, 384), 0), out=buf518)
        del arg259_1
        buf519 = reinterpret_tensor(buf518, (8, 197, 1536), (302592, 1536, 1), 0); del buf518  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf519, arg260_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg260_1
        buf520 = reinterpret_tensor(buf517, (1576, 384), (384, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf519, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg261_1, (1536, 384), (1, 1536), 0), out=buf520)
        del arg261_1
        buf522 = reinterpret_tensor(buf495, (1568, 384), (384, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf521, (1568, 384), (384, 1), 0), reinterpret_tensor(arg277_1, (384, 384), (1, 384), 0), out=buf522)
        del arg277_1
        buf523 = reinterpret_tensor(buf520, (8, 197, 384), (75648, 384, 1), 0); del buf520  # reuse
        buf527 = buf467; del buf467  # reuse
        # Source Nodes: [cat_15, l__mod___blocks_9_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf523, buf486, arg262_1, buf522, arg278_1, arg279_1, arg280_1, buf527, 1576, 384, grid=grid(1576), stream=stream0)
        del arg262_1
        del arg278_1
        del arg279_1
        del arg280_1
        buf528 = buf474; del buf474  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (1576, 384), (384, 1), 0), reinterpret_tensor(arg281_1, (384, 768), (1, 384), 0), out=buf528)
        del arg281_1
        buf529 = reinterpret_tensor(buf486, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf486  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf528, buf529, 605184, grid=grid(605184), stream=stream0)
        buf530 = reinterpret_tensor(buf475, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf475  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf528, buf530, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf531 = reinterpret_tensor(buf481, (48, 197, 197), (38809, 197, 1), 0); del buf481  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf529, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf530, (48, 64, 197), (12608, 197, 1), 0), out=buf531)
        buf535 = reinterpret_tensor(buf477, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf477  # reuse
        # Source Nodes: [attn_57, attn_58], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf531, buf535, 9456, 197, grid=grid(9456), stream=stream0)
        buf534 = reinterpret_tensor(buf530, (1576, 384), (384, 1), 0); del buf530  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf527, (1576, 384), (384, 1), 0), reinterpret_tensor(arg282_1, (384, 384), (1, 384), 0), out=buf534)
        del arg282_1
        buf536 = reinterpret_tensor(buf527, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf527  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf534, buf536, 605184, grid=grid(605184), stream=stream0)
        buf537 = reinterpret_tensor(buf534, (48, 197, 64), (12608, 64, 1), 0); del buf534  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf535, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf536, (48, 197, 64), (12608, 64, 1), 0), out=buf537)
        buf538 = reinterpret_tensor(buf536, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf536  # reuse
        # Source Nodes: [x_176], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf537, buf538, 605184, grid=grid(605184), stream=stream0)
        buf539 = reinterpret_tensor(buf537, (1576, 384), (384, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf538, (1576, 384), (384, 1), 0), reinterpret_tensor(arg283_1, (384, 384), (1, 384), 0), out=buf539)
        del arg283_1
        buf570 = reinterpret_tensor(buf538, (8, 197, 384), (75648, 384, 1), 0); del buf538  # reuse
        # Source Nodes: [l__mod___blocks_9_norm_mlp, patch_embed_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf523, buf539, arg284_1, arg285_1, arg286_1, buf570, 1576, 384, grid=grid(1576), stream=stream0)
        del arg285_1
        del arg286_1
        buf547 = buf494; del buf494  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (25088, 24), (24, 1), 0), reinterpret_tensor(arg293_1, (24, 48), (1, 24), 0), out=buf547)
        del arg293_1
        buf548 = reinterpret_tensor(buf522, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf522  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf547, buf548, 602112, grid=grid(602112), stream=stream0)
        buf549 = reinterpret_tensor(buf521, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf521  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf547, buf549, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf550 = reinterpret_tensor(buf501, (6272, 16, 16), (256, 16, 1), 0); del buf501  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf548, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf549, (6272, 6, 16), (96, 16, 1), 0), out=buf550)
        buf554 = reinterpret_tensor(buf497, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf497  # reuse
        # Source Nodes: [attn_60, attn_61], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf550, buf554, 100352, 16, grid=grid(100352), stream=stream0)
        buf553 = reinterpret_tensor(buf549, (25088, 24), (24, 1), 0); del buf549  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (25088, 24), (24, 1), 0), reinterpret_tensor(arg294_1, (24, 24), (1, 24), 0), out=buf553)
        del arg294_1
        buf555 = reinterpret_tensor(buf546, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf546  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf553, buf555, 602112, grid=grid(602112), stream=stream0)
        buf556 = reinterpret_tensor(buf553, (6272, 16, 6), (96, 6, 1), 0); del buf553  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf554, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf555, (6272, 16, 6), (96, 6, 1), 0), out=buf556)
        buf557 = reinterpret_tensor(buf555, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf555  # reuse
        # Source Nodes: [x_185], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf556, buf557, 602112, grid=grid(602112), stream=stream0)
        buf558 = reinterpret_tensor(buf556, (25088, 24), (24, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (25088, 24), (24, 1), 0), reinterpret_tensor(arg295_1, (24, 24), (1, 24), 0), out=buf558)
        del arg295_1
        buf559 = reinterpret_tensor(buf558, (1568, 16, 24), (384, 24, 1), 0); del buf558  # reuse
        buf563 = reinterpret_tensor(buf557, (1568, 16, 24), (384, 24, 1), 0); del buf557  # reuse
        # Source Nodes: [l__mod___blocks_10_norm_mlp_in, pixel_embed_30, pixel_embed_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf559, buf506, buf513, arg274_1, arg296_1, arg297_1, arg298_1, buf563, 25088, 24, grid=grid(25088), stream=stream0)
        del arg274_1
        del arg296_1
        del arg297_1
        del arg298_1
        buf564 = reinterpret_tensor(buf512, (25088, 96), (96, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf563, (25088, 24), (24, 1), 0), reinterpret_tensor(arg299_1, (24, 96), (1, 24), 0), out=buf564)
        del arg299_1
        buf565 = reinterpret_tensor(buf564, (1568, 16, 96), (1536, 96, 1), 0); del buf564  # reuse
        # Source Nodes: [x_189], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf565, arg300_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg300_1
        buf566 = reinterpret_tensor(buf563, (25088, 24), (24, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf565, (25088, 96), (96, 1), 0), reinterpret_tensor(arg301_1, (96, 24), (1, 96), 0), out=buf566)
        del arg301_1
        buf575 = reinterpret_tensor(buf513, (1568, 16, 24), (384, 24, 1), 0); del buf513  # reuse
        buf600 = buf506; del buf506  # reuse
        # Source Nodes: [l__mod___blocks_10_norm1_proj, l__mod___blocks_11_norm_in, pixel_embed_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf559, buf566, arg302_1, arg303_1, arg304_1, arg319_1, arg320_1, buf575, buf600, 25088, 24, grid=grid(25088), stream=stream0)
        del arg303_1
        del arg304_1
        del arg319_1
        del arg320_1
        buf571 = reinterpret_tensor(buf519, (1576, 1536), (1536, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf570, (1576, 384), (384, 1), 0), reinterpret_tensor(arg287_1, (384, 1536), (1, 384), 0), out=buf571)
        del arg287_1
        buf572 = reinterpret_tensor(buf571, (8, 197, 1536), (302592, 1536, 1), 0); del buf571  # reuse
        # Source Nodes: [x_180], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf572, arg288_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg288_1
        buf573 = reinterpret_tensor(buf570, (1576, 384), (384, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg289_1, (1536, 384), (1, 1536), 0), out=buf573)
        del arg289_1
        buf574 = reinterpret_tensor(buf573, (8, 197, 384), (75648, 384, 1), 0); del buf573  # reuse
        # Source Nodes: [patch_embed_41, patch_embed_43], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf574, buf523, buf539, arg284_1, arg290_1, 605184, grid=grid(605184), stream=stream0)
        del arg284_1
        del arg290_1
        buf576 = reinterpret_tensor(buf548, (1568, 384), (384, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf575, (1568, 384), (384, 1), 0), reinterpret_tensor(arg305_1, (384, 384), (1, 384), 0), out=buf576)
        del arg305_1
        buf580 = reinterpret_tensor(buf539, (8, 197, 384), (75648, 384, 1), 0); del buf539  # reuse
        # Source Nodes: [cat_14, l__mod___blocks_10_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf574, buf576, arg306_1, arg307_1, arg308_1, buf580, 1576, 384, grid=grid(1576), stream=stream0)
        del arg307_1
        del arg308_1
        buf581 = buf528; del buf528  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf580, (1576, 384), (384, 1), 0), reinterpret_tensor(arg309_1, (384, 768), (1, 384), 0), out=buf581)
        del arg309_1
        buf582 = reinterpret_tensor(buf523, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf523  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf581, buf582, 605184, grid=grid(605184), stream=stream0)
        buf583 = reinterpret_tensor(buf529, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf529  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf581, buf583, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf584 = reinterpret_tensor(buf535, (48, 197, 197), (38809, 197, 1), 0); del buf535  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf582, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf583, (48, 64, 197), (12608, 197, 1), 0), out=buf584)
        buf588 = reinterpret_tensor(buf531, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf531  # reuse
        # Source Nodes: [attn_63, attn_64], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf584, buf588, 9456, 197, grid=grid(9456), stream=stream0)
        buf587 = reinterpret_tensor(buf583, (1576, 384), (384, 1), 0); del buf583  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf580, (1576, 384), (384, 1), 0), reinterpret_tensor(arg310_1, (384, 384), (1, 384), 0), out=buf587)
        del arg310_1
        buf589 = reinterpret_tensor(buf580, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf580  # reuse
        # Source Nodes: [matmul_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf587, buf589, 605184, grid=grid(605184), stream=stream0)
        buf590 = reinterpret_tensor(buf587, (48, 197, 64), (12608, 64, 1), 0); del buf587  # reuse
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf589, (48, 197, 64), (12608, 64, 1), 0), out=buf590)
        buf591 = reinterpret_tensor(buf589, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf589  # reuse
        # Source Nodes: [x_194], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf590, buf591, 605184, grid=grid(605184), stream=stream0)
        buf592 = reinterpret_tensor(buf590, (1576, 384), (384, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf591, (1576, 384), (384, 1), 0), reinterpret_tensor(arg311_1, (384, 384), (1, 384), 0), out=buf592)
        del arg311_1
        buf593 = reinterpret_tensor(buf592, (8, 197, 384), (75648, 384, 1), 0); del buf592  # reuse
        buf624 = reinterpret_tensor(buf591, (8, 197, 384), (75648, 384, 1), 0); del buf591  # reuse
        # Source Nodes: [cat_14, l__mod___blocks_10_norm_mlp, patch_embed_45], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf593, buf574, buf576, arg306_1, arg312_1, arg313_1, arg314_1, buf624, 1576, 384, grid=grid(1576), stream=stream0)
        del arg306_1
        del arg312_1
        del arg313_1
        del arg314_1
        buf601 = buf547; del buf547  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (25088, 24), (24, 1), 0), reinterpret_tensor(arg321_1, (24, 48), (1, 24), 0), out=buf601)
        del arg321_1
        buf602 = reinterpret_tensor(buf576, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf576  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf601, buf602, 602112, grid=grid(602112), stream=stream0)
        buf603 = reinterpret_tensor(buf575, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf575  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf601, buf603, 37632, 16, grid=grid(37632, 16), stream=stream0)
        del buf601
        buf604 = reinterpret_tensor(buf554, (6272, 16, 16), (256, 16, 1), 0); del buf554  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf602, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf603, (6272, 6, 16), (96, 16, 1), 0), out=buf604)
        del buf602
        buf608 = reinterpret_tensor(buf550, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf550  # reuse
        # Source Nodes: [attn_66, attn_67], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf604, buf608, 100352, 16, grid=grid(100352), stream=stream0)
        del buf604
        buf607 = reinterpret_tensor(buf603, (25088, 24), (24, 1), 0); del buf603  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (25088, 24), (24, 1), 0), reinterpret_tensor(arg322_1, (24, 24), (1, 24), 0), out=buf607)
        del arg322_1
        buf609 = reinterpret_tensor(buf600, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf600  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf607, buf609, 602112, grid=grid(602112), stream=stream0)
        buf610 = reinterpret_tensor(buf607, (6272, 16, 6), (96, 6, 1), 0); del buf607  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf608, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf609, (6272, 16, 6), (96, 6, 1), 0), out=buf610)
        del buf608
        buf611 = reinterpret_tensor(buf609, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf609  # reuse
        # Source Nodes: [x_203], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf610, buf611, 602112, grid=grid(602112), stream=stream0)
        buf612 = reinterpret_tensor(buf610, (25088, 24), (24, 1), 0); del buf610  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf611, (25088, 24), (24, 1), 0), reinterpret_tensor(arg323_1, (24, 24), (1, 24), 0), out=buf612)
        del arg323_1
        buf613 = reinterpret_tensor(buf612, (1568, 16, 24), (384, 24, 1), 0); del buf612  # reuse
        buf617 = reinterpret_tensor(buf611, (1568, 16, 24), (384, 24, 1), 0); del buf611  # reuse
        # Source Nodes: [l__mod___blocks_11_norm_mlp_in, pixel_embed_33, pixel_embed_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf613, buf559, buf566, arg302_1, arg324_1, arg325_1, arg326_1, buf617, 25088, 24, grid=grid(25088), stream=stream0)
        del arg302_1
        del arg324_1
        del arg325_1
        del arg326_1
        del buf559
        buf618 = reinterpret_tensor(buf565, (25088, 96), (96, 1), 0); del buf565  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf617, (25088, 24), (24, 1), 0), reinterpret_tensor(arg327_1, (24, 96), (1, 24), 0), out=buf618)
        del arg327_1
        buf619 = reinterpret_tensor(buf618, (1568, 16, 96), (1536, 96, 1), 0); del buf618  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf619, arg328_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg328_1
        buf620 = reinterpret_tensor(buf617, (25088, 24), (24, 1), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf619, (25088, 96), (96, 1), 0), reinterpret_tensor(arg329_1, (96, 24), (1, 96), 0), out=buf620)
        del arg329_1
        del buf619
        buf628 = reinterpret_tensor(buf566, (1568, 16, 24), (384, 24, 1), 0); del buf566  # reuse
        # Source Nodes: [l__mod___blocks_11_norm1_proj, pixel_embed_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf613, buf620, arg330_1, arg331_1, arg332_1, buf628, 25088, 24, grid=grid(25088), stream=stream0)
        del arg330_1
        del arg331_1
        del arg332_1
        del buf613
        buf625 = reinterpret_tensor(buf572, (1576, 1536), (1536, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf624, (1576, 384), (384, 1), 0), reinterpret_tensor(arg315_1, (384, 1536), (1, 384), 0), out=buf625)
        del arg315_1
        buf626 = reinterpret_tensor(buf625, (8, 197, 1536), (302592, 1536, 1), 0); del buf625  # reuse
        # Source Nodes: [x_198], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf626, arg316_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg316_1
        buf627 = reinterpret_tensor(buf624, (1576, 384), (384, 1), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf626, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg317_1, (1536, 384), (1, 1536), 0), out=buf627)
        del arg317_1
        buf629 = reinterpret_tensor(buf620, (1568, 384), (384, 1), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf628, (1568, 384), (384, 1), 0), reinterpret_tensor(arg333_1, (384, 384), (1, 384), 0), out=buf629)
        del arg333_1
        del buf628
        buf630 = reinterpret_tensor(buf627, (8, 197, 384), (75648, 384, 1), 0); del buf627  # reuse
        buf634 = buf574; del buf574  # reuse
        # Source Nodes: [cat_13, l__mod___blocks_11_norm_out], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf630, buf593, arg318_1, buf629, arg334_1, arg335_1, arg336_1, buf634, 1576, 384, grid=grid(1576), stream=stream0)
        del arg318_1
        del arg334_1
        del arg335_1
        del arg336_1
        del buf629
        buf635 = buf581; del buf581  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf634, (1576, 384), (384, 1), 0), reinterpret_tensor(arg337_1, (384, 768), (1, 384), 0), out=buf635)
        del arg337_1
        buf636 = reinterpret_tensor(buf593, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf593  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf635, buf636, 605184, grid=grid(605184), stream=stream0)
        buf637 = reinterpret_tensor(buf582, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf582  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf635, buf637, 3072, 197, grid=grid(3072, 197), stream=stream0)
        del buf635
        buf638 = reinterpret_tensor(buf588, (48, 197, 197), (38809, 197, 1), 0); del buf588  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf636, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf637, (48, 64, 197), (12608, 197, 1), 0), out=buf638)
        del buf636
        buf642 = reinterpret_tensor(buf584, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf584  # reuse
        # Source Nodes: [attn_69, attn_70], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_18.run(buf638, buf642, 9456, 197, grid=grid(9456), stream=stream0)
        del buf638
        buf641 = reinterpret_tensor(buf637, (1576, 384), (384, 1), 0); del buf637  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf634, (1576, 384), (384, 1), 0), reinterpret_tensor(arg338_1, (384, 384), (1, 384), 0), out=buf641)
        del arg338_1
        buf643 = reinterpret_tensor(buf634, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf634  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf641, buf643, 605184, grid=grid(605184), stream=stream0)
        buf644 = reinterpret_tensor(buf641, (48, 197, 64), (12608, 64, 1), 0); del buf641  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf642, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf643, (48, 197, 64), (12608, 64, 1), 0), out=buf644)
        del buf642
        buf645 = reinterpret_tensor(buf643, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf643  # reuse
        # Source Nodes: [x_212], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf644, buf645, 605184, grid=grid(605184), stream=stream0)
        buf646 = reinterpret_tensor(buf644, (1576, 384), (384, 1), 0); del buf644  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf645, (1576, 384), (384, 1), 0), reinterpret_tensor(arg339_1, (384, 384), (1, 384), 0), out=buf646)
        del arg339_1
        buf650 = reinterpret_tensor(buf645, (8, 197, 384), (75648, 384, 1), 0); del buf645  # reuse
        # Source Nodes: [l__mod___blocks_11_norm_mlp, patch_embed_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf630, buf646, arg340_1, arg341_1, arg342_1, buf650, 1576, 384, grid=grid(1576), stream=stream0)
        del arg341_1
        del arg342_1
        buf651 = reinterpret_tensor(buf626, (1576, 1536), (1536, 1), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf650, (1576, 384), (384, 1), 0), reinterpret_tensor(arg343_1, (384, 1536), (1, 384), 0), out=buf651)
        del arg343_1
        buf652 = reinterpret_tensor(buf651, (8, 197, 1536), (302592, 1536, 1), 0); del buf651  # reuse
        # Source Nodes: [x_216], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf652, arg344_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg344_1
        buf653 = reinterpret_tensor(buf650, (1576, 384), (384, 1), 0); del buf650  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf652, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg345_1, (1536, 384), (1, 1536), 0), out=buf653)
        del arg345_1
        del buf652
        buf654 = reinterpret_tensor(buf653, (8, 197, 384), (75648, 384, 1), 0); del buf653  # reuse
        buf655 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf656 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        # Source Nodes: [patch_embed_49, patch_embed_51, x_221], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_28.run(buf654, buf630, buf646, arg340_1, arg346_1, buf655, buf656, 1576, 384, grid=grid(1576), stream=stream0)
        del arg340_1
        del arg346_1
        del buf630
        del buf646
        buf658 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf654, buf655, buf656, arg347_1, arg348_1, buf658, 3072, grid=grid(3072), stream=stream0)
        del arg347_1
        del arg348_1
        del buf654
        del buf655
        del buf656
        buf659 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223, x_224], Original ATen: [aten.addmm, aten.clone]
        extern_kernels.addmm(arg350_1, buf658, reinterpret_tensor(arg349_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf659)
        del arg349_1
        del arg350_1
        return (buf659, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 24, 4, 4), (384, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tnt_s_patch16_224', benchmark_compiled_module)
