
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


# kernel path: /tmp/torchinductor_youkaichao/zh/czhx6mxaeabvhufce44xjzymqpqj6rs4xhkl4lilj5zunadw53kx.py
# Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm]
# x_4 => add
# x_6 => clone_1, var_mean
triton_red_fused_add_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 196
    x2 = (xindex // 1176)
    x5 = xindex % 1176
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x6), tmp6, xmask)
    tl.store(out_ptr1 + (x6), tmp7, xmask)
    tl.store(out_ptr2 + (x6), tmp8, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/wc/cwch2qi6b7iz2lcf4fckj3amrsglkziy47k754a7c5ftve5klysb.py
# Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm]
# x_4 => add
# x_6 => clone_1, var_mean
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ek/cek343bpb3kjx5jhfp6zzmgsxcazxxl2qpbslkfuentyudycfrnr.py
# Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm]
# x_4 => add
# x_6 => add_1, add_2, clone_1, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr2 + (x2 + (768*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/34/c34jqdpr6i2yehuqw5w32hjfd52lvkazsh3qo6ckpljcc7ijnlih.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_5
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 196
    x2 = (xindex // 9408) % 16
    x3 = (xindex // 150528)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (1536*x1) + (301056*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6cho4bp64lrvih33o2imkc6et45b5xcszux7w4oresvvoihqsth.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_6
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3zs4ocvhcjcieevlzhc6zqjymohg5eictwmj5ggecsqcs73kj5.py
# Source Nodes: [patch_score, patch_score_1], Original ATen: [aten._softmax, aten.mul]
# patch_score => mul_2
# patch_score_1 => amax, exp, sub_2, sum_1
triton_per_fused__softmax_mul_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.14433756729740643
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h7/ch72ggr4ewl356cvf3zt5zxiwritmifsssmkm6inipvvresomfnd.py
# Source Nodes: [rel_indices, setitem, setitem_1, setitem_2, to], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
# rel_indices => full
# setitem => copy, select_scatter
# setitem_1 => copy_1, select_scatter_1
# setitem_2 => copy_2, select_scatter_2
# to => device_put
triton_poi_fused__to_copy_copy_select_scatter_zeros_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_select_scatter_zeros_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = ((-1)*(x2 % 14)) + (x1 % 14)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full([1], 1, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = ((-1)*(x2 // 14)) + (x1 // 14)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 2, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = ((x1 // 14)*(x1 // 14)) + ((x2 // 14)*(x2 // 14)) + ((x1 % 14)*(x1 % 14)) + ((x2 % 14)*(x2 % 14)) + ((-2)*(x1 // 14)*(x2 // 14)) + ((-2)*(x1 % 14)*(x2 % 14))
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.0
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tl.where(tmp6, tmp8, tmp14)
    tmp16 = tl.where(tmp2, tmp4, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crcfycmgce2mgxa4qs7nvcwy3gf6ntbhhz76ltbvx3o5hugekggi.py
# Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten.clone]
# l__mod___blocks_0_attn_pos_proj => clone_4
triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 115248
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5fe7xmei7zwwxtdyxjbkcfazbiaspujcfisxcskjdtycsyy2my.py
# Source Nodes: [pos_score_2], Original ATen: [aten._softmax]
# pos_score_2 => amax_1, clone_7, exp_1, sub_3, sum_2
triton_red_fused__softmax_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + (x0 + (16*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 + tmp1
        tmp8 = tmp7 - tmp4
        tmp9 = tl.exp(tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2l6zcznhtwj244nk645yt3tgm2hhdeghfhtgy2dsgyxbscxdnvg.py
# Source Nodes: [attn, mul_1, mul_2, patch_score, patch_score_1, pos_score_2, sigmoid, sigmoid_1, sub_1], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
# attn => add_5
# mul_1 => mul_3
# mul_2 => mul_4
# patch_score => mul_2
# patch_score_1 => div, exp, sub_2
# pos_score_2 => clone_7, div_1, exp_1, sub_3
# sigmoid => sigmoid
# sigmoid_1 => sigmoid_1
# sub_1 => sub_4
triton_poi_fused__softmax_add_mul_rsub_sigmoid_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_mul_rsub_sigmoid_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 38416
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 16
    x5 = xindex
    y4 = yindex
    x3 = (xindex // 196)
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x5 + (38416*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x3 + (196*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x3 + (196*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0 + (16*x5) + (614656*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0 + (16*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (y0 + (16*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp5 = 0.14433756729740643
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp11 = tmp9 / tmp10
    tmp12 = tmp3 * tmp11
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tl.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tmp12 + tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (38416*y4)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2a3fysmcaa24ioq7q4x4unx6urgaomqjahgeyrnk5i2wi5pm3qe.py
# Source Nodes: [attn_1, sum_1], Original ATen: [aten.div, aten.sum]
# attn_1 => div_2
# sum_1 => sum_3
triton_per_fused_div_sum_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp0 / tmp4
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp5, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c42ficdqtyv3x6ipmdr5rtp5nwqbo52tjy546qldczwcv2wtp4af.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_9
triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 196
    x2 = (xindex // 9408) % 16
    x3 = (xindex // 150528)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (768*x1) + (150528*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5idprvjxxfwmny3s3jkum3534x7qgtorzm6b64een5w6p6nyfsw.py
# Source Nodes: [x_7], Original ATen: [aten.clone]
# x_7 => clone_10
triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 16
    x2 = (xindex // 768) % 196
    x3 = (xindex // 150528)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (9408*x1) + (150528*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3h5gltryblmsutdj6vb2orthquqjr65pv26uup63yyzr2nnm3tu.py
# Source Nodes: [x_10, x_11, x_4], Original ATen: [aten.add, aten.native_layer_norm]
# x_10 => add_6
# x_11 => add_7, add_8, clone_12, mul_5, mul_6, rsqrt_1, sub_5, var_mean_1
# x_4 => add
triton_per_fused_add_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (150528*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hw/chwfwgw4mhm5wo5flo7ui4l6cm6fxwqs3vqtz2ivsjur2kgbxug4.py
# Source Nodes: [x_13], Original ATen: [aten.gelu]
# x_13 => add_9, erf, mul_7, mul_8, mul_9
triton_poi_fused_gelu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
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


# kernel path: /tmp/torchinductor_youkaichao/lx/clxwrfsaigna63hvrpaigemgd2dy5tmyf4rofdq24mrfy7uwtqif.py
# Source Nodes: [x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm]
# x_19 => add_10
# x_20 => add_11, add_12, clone_15, mul_10, mul_11, rsqrt_2, sub_6, var_mean_2
triton_per_fused_add_native_layer_norm_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gk7rafncpsfy7rkb6b7m3vq34petpug3dnviq6qlp5yxa3fh4o.py
# Source Nodes: [x_19, x_24, x_25], Original ATen: [aten.add, aten.native_layer_norm]
# x_19 => add_10
# x_24 => add_16
# x_25 => add_17, add_18, clone_26, mul_15, mul_16, rsqrt_3, sub_11, var_mean_3
triton_per_fused_add_native_layer_norm_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2v7acjzalmdz33hwwoiwffed4n5orjjr5kxd3ujejxldxddjtw5.py
# Source Nodes: [cat_1, x_147], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_1 => cat
# x_147 => add_101, add_102, mul_100, mul_101, rsqrt_20, sub_60, var_mean_20
triton_per_fused_cat_native_layer_norm_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-768) + r2 + (768*x0) + (150528*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-768) + r2 + (768*x0) + (150528*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 768.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmit5sff3bdx32vchfg6ikdbmqkrtjws42byozvrzecxom6quzux.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_141
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 197
    x2 = (xindex // 9456) % 16
    x3 = (xindex // 151296)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (2304*x1) + (453888*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7srmhpinieiglizvdgcsy7fcfyxs4kb4qjoseuupflxhfwm2fbb.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_142
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 197
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
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (2304*x2) + (453888*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nv/cnv4gwyhyje3loxjtotvqvmxqomk4cbggypqs3d2jwuwfrx74e3u.py
# Source Nodes: [attn_40, attn_41], Original ATen: [aten._softmax, aten.mul]
# attn_40 => mul_102
# attn_41 => amax_20, div_30, exp_20, sub_61, sum_31
triton_per_fused__softmax_mul_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25216
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
    tmp1 = 0.14433756729740643
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


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3xhbqnv3eke7bs76jlcvhhhrmrkq3cfhj7s6tvwsyol5m7hri2.py
# Source Nodes: [matmul_21], Original ATen: [aten.clone]
# matmul_21 => clone_144
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 197
    x2 = (xindex // 9456) % 16
    x3 = (xindex // 151296)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + (48*x2) + (2304*x1) + (453888*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6fh6a23a4ak4tm5sfnbnmy2gtxpkdkj7ucouiadp7ian7w3bfi.py
# Source Nodes: [x_148], Original ATen: [aten.clone]
# x_148 => clone_145
triton_poi_fused_clone_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 16
    x2 = (xindex // 768) % 197
    x3 = (xindex // 151296)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (9456*x1) + (151296*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2x27t2eyoffvb4nutjp5k5by6m25rawg3ah5s2cpk2v37mjakc.py
# Source Nodes: [cat_1, x_151, x_152], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# x_151 => add_103
# x_152 => add_104, add_105, mul_103, mul_104, rsqrt_21, sub_62, var_mean_21
triton_per_fused_add_cat_native_layer_norm_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp19 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-768) + r2 + (768*x0) + (150528*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-768) + r2 + (768*x0) + (150528*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp30 = tl.full([1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjf3q6ks7ovf23vekhi3ueyphbvl4du2bvomwzwn7tiwsi55sesm.py
# Source Nodes: [x_154], Original ATen: [aten.gelu]
# x_154 => add_106, erf_10, mul_105, mul_106, mul_107
triton_poi_fused_gelu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4841472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
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


# kernel path: /tmp/torchinductor_youkaichao/xu/cxu5vt4z2fcsrxmph3oyw6cnka2u27mtqwzswaulkzgvl3w7tfjb.py
# Source Nodes: [x_160, x_161], Original ATen: [aten.add, aten.native_layer_norm]
# x_160 => add_107
# x_161 => add_108, add_109, mul_108, mul_109, rsqrt_22, sub_63, var_mean_22
triton_per_fused_add_native_layer_norm_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jddnem6eanbsimndz5ggwhstnph2el7ipwqqidowzos74hq6yx.py
# Source Nodes: [x_160, x_165, x_166], Original ATen: [aten.add, aten.native_layer_norm]
# x_160 => add_107
# x_165 => add_110
# x_166 => add_111, add_112, mul_111, mul_112, rsqrt_23, sub_65, var_mean_23
triton_per_fused_add_native_layer_norm_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckylbxoqq22mz4wa5wqijcuxjwjsjz7uz7y6hkzx3h7v4lps7m4d.py
# Source Nodes: [x_174, x_177], Original ATen: [aten.add, aten.native_layer_norm]
# x_174 => add_114
# x_177 => var_mean_24
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curhcx3nztkcjc7rrgdelb4v75z2hp2b7r2tiu4vfovydxik5b5r.py
# Source Nodes: [x_179], Original ATen: [aten.clone]
# x_179 => clone_157
triton_poi_fused_clone_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (151296*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (151296*x1)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (197*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (197*x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 196, 768), (150528, 768, 1))
    assert_size_stride(arg1_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (16, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (16, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (16, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (16, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (16, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (16, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (16, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (1536, 768), (768, 1))
    assert_size_stride(arg65_1, (16, 3), (3, 1))
    assert_size_stride(arg66_1, (16, ), (1, ))
    assert_size_stride(arg67_1, (768, 768), (768, 1))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (3072, 768), (768, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (1536, 768), (768, 1))
    assert_size_stride(arg75_1, (16, 3), (3, 1))
    assert_size_stride(arg76_1, (16, ), (1, ))
    assert_size_stride(arg77_1, (768, 768), (768, 1))
    assert_size_stride(arg78_1, (768, 768), (768, 1))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (3072, 768), (768, 1))
    assert_size_stride(arg81_1, (3072, ), (1, ))
    assert_size_stride(arg82_1, (768, 3072), (3072, 1))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (1536, 768), (768, 1))
    assert_size_stride(arg85_1, (16, 3), (3, 1))
    assert_size_stride(arg86_1, (16, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (3072, 768), (768, 1))
    assert_size_stride(arg91_1, (3072, ), (1, ))
    assert_size_stride(arg92_1, (768, 3072), (3072, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (1536, 768), (768, 1))
    assert_size_stride(arg95_1, (16, 3), (3, 1))
    assert_size_stride(arg96_1, (16, ), (1, ))
    assert_size_stride(arg97_1, (768, 768), (768, 1))
    assert_size_stride(arg98_1, (768, 768), (768, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (3072, 768), (768, 1))
    assert_size_stride(arg101_1, (3072, ), (1, ))
    assert_size_stride(arg102_1, (768, 3072), (3072, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (1536, 768), (768, 1))
    assert_size_stride(arg105_1, (16, 3), (3, 1))
    assert_size_stride(arg106_1, (16, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (3072, 768), (768, 1))
    assert_size_stride(arg111_1, (3072, ), (1, ))
    assert_size_stride(arg112_1, (768, 3072), (3072, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (1536, 768), (768, 1))
    assert_size_stride(arg115_1, (16, 3), (3, 1))
    assert_size_stride(arg116_1, (16, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (3072, 768), (768, 1))
    assert_size_stride(arg121_1, (3072, ), (1, ))
    assert_size_stride(arg122_1, (768, 3072), (3072, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (1536, 768), (768, 1))
    assert_size_stride(arg125_1, (16, 3), (3, 1))
    assert_size_stride(arg126_1, (16, ), (1, ))
    assert_size_stride(arg127_1, (768, 768), (768, 1))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (3072, 768), (768, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (768, 3072), (3072, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (1536, 768), (768, 1))
    assert_size_stride(arg135_1, (16, 3), (3, 1))
    assert_size_stride(arg136_1, (16, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (3072, 768), (768, 1))
    assert_size_stride(arg141_1, (3072, ), (1, ))
    assert_size_stride(arg142_1, (768, 3072), (3072, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (1536, 768), (768, 1))
    assert_size_stride(arg145_1, (16, 3), (3, 1))
    assert_size_stride(arg146_1, (16, ), (1, ))
    assert_size_stride(arg147_1, (768, 768), (768, 1))
    assert_size_stride(arg148_1, (768, 768), (768, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (3072, 768), (768, 1))
    assert_size_stride(arg151_1, (3072, ), (1, ))
    assert_size_stride(arg152_1, (768, 3072), (3072, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (1536, 768), (768, 1))
    assert_size_stride(arg155_1, (16, 3), (3, 1))
    assert_size_stride(arg156_1, (16, ), (1, ))
    assert_size_stride(arg157_1, (768, 768), (768, 1))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (3072, 768), (768, 1))
    assert_size_stride(arg161_1, (3072, ), (1, ))
    assert_size_stride(arg162_1, (768, 3072), (3072, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (2304, 768), (768, 1))
    assert_size_stride(arg165_1, (768, 768), (768, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (3072, 768), (768, 1))
    assert_size_stride(arg168_1, (3072, ), (1, ))
    assert_size_stride(arg169_1, (768, 3072), (3072, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (2304, 768), (768, 1))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (3072, 768), (768, 1))
    assert_size_stride(arg175_1, (3072, ), (1, ))
    assert_size_stride(arg176_1, (768, 3072), (3072, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (1000, 768), (768, 1))
    assert_size_stride(arg179_1, (1000, ), (1, ))
    assert_size_stride(arg180_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg180_1, arg62_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg180_1
        del arg62_1
        buf1 = empty_strided((8, 196, 1, 6), (1176, 6, 9408, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 6), (1176, 6, 9408, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 6), (1176, 6, 9408, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_native_layer_norm_0.run(buf0, arg63_1, arg0_1, buf1, buf2, buf3, 9408, 128, grid=grid(9408), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 6, grid=grid(1568), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_2.run(buf0, arg63_1, arg0_1, buf4, buf5, arg2_1, arg3_1, buf7, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg2_1
        del arg3_1
        del buf4
        del buf5
        buf8 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 1536), (1, 768), 0), out=buf8)
        del arg64_1
        buf9 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf8, buf9, 1204224, grid=grid(1204224), stream=stream0)
        buf10 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf8, buf10, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf11 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf10, (128, 48, 196), (9408, 196, 1), 0), out=buf11)
        buf12 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [patch_score, patch_score_1], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf11, buf12, buf13, 25088, 196, grid=grid(25088), stream=stream0)
        buf14 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices, setitem, setitem_1, setitem_2, to], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf14, 115248, grid=grid(115248), stream=stream0)
        buf15 = empty((8, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf14, buf15, 921984, grid=grid(921984), stream=stream0)
        buf16 = empty((307328, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (307328, 3), (3, 1), 0), reinterpret_tensor(arg65_1, (3, 16), (1, 3), 0), out=buf16)
        del arg65_1
        buf17 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 25088), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf16, arg66_1, buf17, buf18, 25088, 196, grid=grid(25088), stream=stream0)
        buf19 = reinterpret_tensor(buf11, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf11  # reuse
        # Source Nodes: [attn, mul_1, mul_2, patch_score, patch_score_1, pos_score_2, sigmoid, sigmoid_1, sub_1], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf19, arg4_1, buf12, buf13, buf16, arg66_1, buf17, buf18, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg4_1
        del arg66_1
        buf22 = reinterpret_tensor(buf16, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf16  # reuse
        # Source Nodes: [attn_1, sum_1], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf19, buf22, 25088, 196, grid=grid(25088), stream=stream0)
        buf21 = reinterpret_tensor(buf9, (1568, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [l__mod___blocks_0_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 768), (1, 768), 0), out=buf21)
        del arg67_1
        buf23 = reinterpret_tensor(buf7, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf7  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf21, buf23, 1204224, grid=grid(1204224), stream=stream0)
        buf24 = reinterpret_tensor(buf21, (128, 196, 48), (9408, 48, 1), 0); del buf21  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf22, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf23, (128, 196, 48), (9408, 48, 1), 0), out=buf24)
        buf25 = reinterpret_tensor(buf23, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf23  # reuse
        # Source Nodes: [x_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf24, buf25, 1204224, grid=grid(1204224), stream=stream0)
        buf26 = reinterpret_tensor(buf24, (1568, 768), (768, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (1568, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), out=buf26)
        del arg68_1
        buf27 = reinterpret_tensor(buf26, (8, 196, 768), (150528, 768, 1), 0); del buf26  # reuse
        buf31 = reinterpret_tensor(buf25, (8, 196, 768), (150528, 768, 1), 0); del buf25  # reuse
        # Source Nodes: [x_10, x_11, x_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf27, buf0, arg63_1, arg0_1, arg69_1, arg5_1, arg6_1, buf31, 1568, 768, grid=grid(1568), stream=stream0)
        del arg0_1
        del arg5_1
        del arg63_1
        del arg69_1
        del arg6_1
        buf32 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (1568, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 3072), (1, 768), 0), out=buf32)
        del arg70_1
        buf33 = reinterpret_tensor(buf32, (8, 196, 3072), (602112, 3072, 1), 0); del buf32  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf33, arg71_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg71_1
        buf34 = reinterpret_tensor(buf31, (1568, 768), (768, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg72_1, (3072, 768), (1, 3072), 0), out=buf34)
        del arg72_1
        buf38 = reinterpret_tensor(buf0, (8, 196, 768), (150528, 768, 1), 0); del buf0  # reuse
        # Source Nodes: [x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf27, buf34, arg73_1, arg7_1, arg8_1, buf38, 1568, 768, grid=grid(1568), stream=stream0)
        del arg7_1
        del arg8_1
        buf39 = buf8; del buf8  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (1568, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 1536), (1, 768), 0), out=buf39)
        del arg74_1
        buf40 = reinterpret_tensor(buf10, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf10  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf39, buf40, 1204224, grid=grid(1204224), stream=stream0)
        buf41 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf39, buf41, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf42 = reinterpret_tensor(buf22, (128, 196, 196), (38416, 196, 1), 0); del buf22  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf40, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf41, (128, 48, 196), (9408, 196, 1), 0), out=buf42)
        buf43 = reinterpret_tensor(buf18, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf18  # reuse
        buf44 = reinterpret_tensor(buf17, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf17  # reuse
        # Source Nodes: [patch_score_2, patch_score_3], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf42, buf43, buf44, 25088, 196, grid=grid(25088), stream=stream0)
        buf45 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_1, setitem_3, setitem_4, setitem_5, to_1], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf45, 115248, grid=grid(115248), stream=stream0)
        buf46 = buf15; del buf15  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf45, buf46, 921984, grid=grid(921984), stream=stream0)
        buf47 = reinterpret_tensor(buf19, (307328, 16), (16, 1), 0); del buf19  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (307328, 3), (3, 1), 0), reinterpret_tensor(arg75_1, (3, 16), (1, 3), 0), out=buf47)
        del arg75_1
        buf48 = reinterpret_tensor(buf13, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf13  # reuse
        buf49 = reinterpret_tensor(buf12, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf12  # reuse
        # Source Nodes: [pos_score_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf47, arg76_1, buf48, buf49, 25088, 196, grid=grid(25088), stream=stream0)
        buf50 = reinterpret_tensor(buf42, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf42  # reuse
        # Source Nodes: [attn_4, mul_4, mul_5, patch_score_2, patch_score_3, pos_score_5, sigmoid_2, sigmoid_3, sub_3], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf50, arg9_1, buf43, buf44, buf47, arg76_1, buf48, buf49, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg76_1
        del arg9_1
        buf53 = reinterpret_tensor(buf47, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf47  # reuse
        # Source Nodes: [attn_5, sum_2], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf50, buf53, 25088, 196, grid=grid(25088), stream=stream0)
        buf52 = reinterpret_tensor(buf41, (1568, 768), (768, 1), 0); del buf41  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (1568, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), out=buf52)
        del arg77_1
        buf54 = reinterpret_tensor(buf38, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf38  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf52, buf54, 1204224, grid=grid(1204224), stream=stream0)
        buf55 = reinterpret_tensor(buf52, (128, 196, 48), (9408, 48, 1), 0); del buf52  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf54, (128, 196, 48), (9408, 48, 1), 0), out=buf55)
        buf56 = reinterpret_tensor(buf54, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf54  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf55, buf56, 1204224, grid=grid(1204224), stream=stream0)
        buf57 = reinterpret_tensor(buf55, (1568, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (1568, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 768), (1, 768), 0), out=buf57)
        del arg78_1
        buf58 = reinterpret_tensor(buf57, (8, 196, 768), (150528, 768, 1), 0); del buf57  # reuse
        buf62 = reinterpret_tensor(buf56, (8, 196, 768), (150528, 768, 1), 0); del buf56  # reuse
        # Source Nodes: [x_19, x_24, x_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf58, buf27, buf34, arg73_1, arg79_1, arg10_1, arg11_1, buf62, 1568, 768, grid=grid(1568), stream=stream0)
        del arg10_1
        del arg11_1
        del arg73_1
        del arg79_1
        buf63 = reinterpret_tensor(buf33, (1568, 3072), (3072, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (1568, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 3072), (1, 768), 0), out=buf63)
        del arg80_1
        buf64 = reinterpret_tensor(buf63, (8, 196, 3072), (602112, 3072, 1), 0); del buf63  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf64, arg81_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg81_1
        buf65 = reinterpret_tensor(buf62, (1568, 768), (768, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), out=buf65)
        del arg82_1
        buf69 = reinterpret_tensor(buf34, (8, 196, 768), (150528, 768, 1), 0); del buf34  # reuse
        # Source Nodes: [x_33, x_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf58, buf65, arg83_1, arg12_1, arg13_1, buf69, 1568, 768, grid=grid(1568), stream=stream0)
        del arg12_1
        del arg13_1
        buf70 = buf39; del buf39  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1568, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 1536), (1, 768), 0), out=buf70)
        del arg84_1
        buf71 = reinterpret_tensor(buf27, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf27  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf70, buf71, 1204224, grid=grid(1204224), stream=stream0)
        buf72 = reinterpret_tensor(buf40, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf40  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf70, buf72, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf73 = reinterpret_tensor(buf53, (128, 196, 196), (38416, 196, 1), 0); del buf53  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf71, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf72, (128, 48, 196), (9408, 196, 1), 0), out=buf73)
        buf74 = reinterpret_tensor(buf49, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf49  # reuse
        buf75 = reinterpret_tensor(buf48, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf48  # reuse
        # Source Nodes: [patch_score_4, patch_score_5], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf73, buf74, buf75, 25088, 196, grid=grid(25088), stream=stream0)
        buf76 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_2, setitem_6, setitem_7, setitem_8, to_2], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf76, 115248, grid=grid(115248), stream=stream0)
        buf77 = buf46; del buf46  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf76, buf77, 921984, grid=grid(921984), stream=stream0)
        buf78 = reinterpret_tensor(buf50, (307328, 16), (16, 1), 0); del buf50  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (307328, 3), (3, 1), 0), reinterpret_tensor(arg85_1, (3, 16), (1, 3), 0), out=buf78)
        del arg85_1
        buf79 = reinterpret_tensor(buf44, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf44  # reuse
        buf80 = reinterpret_tensor(buf43, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf43  # reuse
        # Source Nodes: [pos_score_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf78, arg86_1, buf79, buf80, 25088, 196, grid=grid(25088), stream=stream0)
        buf81 = reinterpret_tensor(buf73, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf73  # reuse
        # Source Nodes: [attn_8, mul_7, mul_8, patch_score_4, patch_score_5, pos_score_8, sigmoid_4, sigmoid_5, sub_5], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf81, arg14_1, buf74, buf75, buf78, arg86_1, buf79, buf80, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg14_1
        del arg86_1
        buf84 = reinterpret_tensor(buf78, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf78  # reuse
        # Source Nodes: [attn_9, sum_3], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf81, buf84, 25088, 196, grid=grid(25088), stream=stream0)
        buf83 = reinterpret_tensor(buf72, (1568, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1568, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf83)
        del arg87_1
        buf85 = reinterpret_tensor(buf69, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf69  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf83, buf85, 1204224, grid=grid(1204224), stream=stream0)
        buf86 = reinterpret_tensor(buf83, (128, 196, 48), (9408, 48, 1), 0); del buf83  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf85, (128, 196, 48), (9408, 48, 1), 0), out=buf86)
        buf87 = reinterpret_tensor(buf85, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf85  # reuse
        # Source Nodes: [x_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf86, buf87, 1204224, grid=grid(1204224), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (1568, 768), (768, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1568, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), out=buf88)
        del arg88_1
        buf89 = reinterpret_tensor(buf88, (8, 196, 768), (150528, 768, 1), 0); del buf88  # reuse
        buf93 = reinterpret_tensor(buf87, (8, 196, 768), (150528, 768, 1), 0); del buf87  # reuse
        # Source Nodes: [x_33, x_38, x_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf89, buf58, buf65, arg83_1, arg89_1, arg15_1, arg16_1, buf93, 1568, 768, grid=grid(1568), stream=stream0)
        del arg15_1
        del arg16_1
        del arg83_1
        del arg89_1
        buf94 = reinterpret_tensor(buf64, (1568, 3072), (3072, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (1568, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 3072), (1, 768), 0), out=buf94)
        del arg90_1
        buf95 = reinterpret_tensor(buf94, (8, 196, 3072), (602112, 3072, 1), 0); del buf94  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf95, arg91_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg91_1
        buf96 = reinterpret_tensor(buf93, (1568, 768), (768, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg92_1, (3072, 768), (1, 3072), 0), out=buf96)
        del arg92_1
        buf100 = reinterpret_tensor(buf65, (8, 196, 768), (150528, 768, 1), 0); del buf65  # reuse
        # Source Nodes: [x_47, x_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf89, buf96, arg93_1, arg17_1, arg18_1, buf100, 1568, 768, grid=grid(1568), stream=stream0)
        del arg17_1
        del arg18_1
        buf101 = buf70; del buf70  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1568, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 1536), (1, 768), 0), out=buf101)
        del arg94_1
        buf102 = reinterpret_tensor(buf58, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf58  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf101, buf102, 1204224, grid=grid(1204224), stream=stream0)
        buf103 = reinterpret_tensor(buf71, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf71  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf101, buf103, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf104 = reinterpret_tensor(buf84, (128, 196, 196), (38416, 196, 1), 0); del buf84  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf102, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf103, (128, 48, 196), (9408, 196, 1), 0), out=buf104)
        buf105 = reinterpret_tensor(buf80, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf80  # reuse
        buf106 = reinterpret_tensor(buf79, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf79  # reuse
        # Source Nodes: [patch_score_6, patch_score_7], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf104, buf105, buf106, 25088, 196, grid=grid(25088), stream=stream0)
        buf107 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_3, setitem_10, setitem_11, setitem_9, to_3], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf107, 115248, grid=grid(115248), stream=stream0)
        buf108 = buf77; del buf77  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf107, buf108, 921984, grid=grid(921984), stream=stream0)
        buf109 = reinterpret_tensor(buf81, (307328, 16), (16, 1), 0); del buf81  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (307328, 3), (3, 1), 0), reinterpret_tensor(arg95_1, (3, 16), (1, 3), 0), out=buf109)
        del arg95_1
        buf110 = reinterpret_tensor(buf75, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf75  # reuse
        buf111 = reinterpret_tensor(buf74, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf74  # reuse
        # Source Nodes: [pos_score_11], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf109, arg96_1, buf110, buf111, 25088, 196, grid=grid(25088), stream=stream0)
        buf112 = reinterpret_tensor(buf104, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf104  # reuse
        # Source Nodes: [attn_12, mul_10, mul_11, patch_score_6, patch_score_7, pos_score_11, sigmoid_6, sigmoid_7, sub_7], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf112, arg19_1, buf105, buf106, buf109, arg96_1, buf110, buf111, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg19_1
        del arg96_1
        buf115 = reinterpret_tensor(buf109, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf109  # reuse
        # Source Nodes: [attn_13, sum_4], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf112, buf115, 25088, 196, grid=grid(25088), stream=stream0)
        buf114 = reinterpret_tensor(buf103, (1568, 768), (768, 1), 0); del buf103  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1568, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 768), (1, 768), 0), out=buf114)
        del arg97_1
        buf116 = reinterpret_tensor(buf100, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf100  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf114, buf116, 1204224, grid=grid(1204224), stream=stream0)
        buf117 = reinterpret_tensor(buf114, (128, 196, 48), (9408, 48, 1), 0); del buf114  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf116, (128, 196, 48), (9408, 48, 1), 0), out=buf117)
        buf118 = reinterpret_tensor(buf116, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf116  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf117, buf118, 1204224, grid=grid(1204224), stream=stream0)
        buf119 = reinterpret_tensor(buf117, (1568, 768), (768, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (1568, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 768), (1, 768), 0), out=buf119)
        del arg98_1
        buf120 = reinterpret_tensor(buf119, (8, 196, 768), (150528, 768, 1), 0); del buf119  # reuse
        buf124 = reinterpret_tensor(buf118, (8, 196, 768), (150528, 768, 1), 0); del buf118  # reuse
        # Source Nodes: [x_47, x_52, x_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf120, buf89, buf96, arg93_1, arg99_1, arg20_1, arg21_1, buf124, 1568, 768, grid=grid(1568), stream=stream0)
        del arg20_1
        del arg21_1
        del arg93_1
        del arg99_1
        buf125 = reinterpret_tensor(buf95, (1568, 3072), (3072, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1568, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 3072), (1, 768), 0), out=buf125)
        del arg100_1
        buf126 = reinterpret_tensor(buf125, (8, 196, 3072), (602112, 3072, 1), 0); del buf125  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf126, arg101_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg101_1
        buf127 = reinterpret_tensor(buf124, (1568, 768), (768, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg102_1, (3072, 768), (1, 3072), 0), out=buf127)
        del arg102_1
        buf131 = reinterpret_tensor(buf96, (8, 196, 768), (150528, 768, 1), 0); del buf96  # reuse
        # Source Nodes: [x_61, x_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf120, buf127, arg103_1, arg22_1, arg23_1, buf131, 1568, 768, grid=grid(1568), stream=stream0)
        del arg22_1
        del arg23_1
        buf132 = buf101; del buf101  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (1568, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 1536), (1, 768), 0), out=buf132)
        del arg104_1
        buf133 = reinterpret_tensor(buf89, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf89  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf132, buf133, 1204224, grid=grid(1204224), stream=stream0)
        buf134 = reinterpret_tensor(buf102, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf102  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf132, buf134, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf135 = reinterpret_tensor(buf115, (128, 196, 196), (38416, 196, 1), 0); del buf115  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf133, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf134, (128, 48, 196), (9408, 196, 1), 0), out=buf135)
        buf136 = reinterpret_tensor(buf111, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf111  # reuse
        buf137 = reinterpret_tensor(buf110, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf110  # reuse
        # Source Nodes: [patch_score_8, patch_score_9], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf135, buf136, buf137, 25088, 196, grid=grid(25088), stream=stream0)
        buf138 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_4, setitem_12, setitem_13, setitem_14, to_4], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf138, 115248, grid=grid(115248), stream=stream0)
        buf139 = buf108; del buf108  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf138, buf139, 921984, grid=grid(921984), stream=stream0)
        buf140 = reinterpret_tensor(buf112, (307328, 16), (16, 1), 0); del buf112  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (307328, 3), (3, 1), 0), reinterpret_tensor(arg105_1, (3, 16), (1, 3), 0), out=buf140)
        del arg105_1
        buf141 = reinterpret_tensor(buf106, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf106  # reuse
        buf142 = reinterpret_tensor(buf105, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf105  # reuse
        # Source Nodes: [pos_score_14], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf140, arg106_1, buf141, buf142, 25088, 196, grid=grid(25088), stream=stream0)
        buf143 = reinterpret_tensor(buf135, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf135  # reuse
        # Source Nodes: [attn_16, mul_13, mul_14, patch_score_8, patch_score_9, pos_score_14, sigmoid_8, sigmoid_9, sub_9], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf143, arg24_1, buf136, buf137, buf140, arg106_1, buf141, buf142, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg106_1
        del arg24_1
        buf146 = reinterpret_tensor(buf140, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf140  # reuse
        # Source Nodes: [attn_17, sum_5], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf143, buf146, 25088, 196, grid=grid(25088), stream=stream0)
        buf145 = reinterpret_tensor(buf134, (1568, 768), (768, 1), 0); del buf134  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (1568, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), out=buf145)
        del arg107_1
        buf147 = reinterpret_tensor(buf131, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf131  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf145, buf147, 1204224, grid=grid(1204224), stream=stream0)
        buf148 = reinterpret_tensor(buf145, (128, 196, 48), (9408, 48, 1), 0); del buf145  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf147, (128, 196, 48), (9408, 48, 1), 0), out=buf148)
        buf149 = reinterpret_tensor(buf147, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf147  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf148, buf149, 1204224, grid=grid(1204224), stream=stream0)
        buf150 = reinterpret_tensor(buf148, (1568, 768), (768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (1568, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), out=buf150)
        del arg108_1
        buf151 = reinterpret_tensor(buf150, (8, 196, 768), (150528, 768, 1), 0); del buf150  # reuse
        buf155 = reinterpret_tensor(buf149, (8, 196, 768), (150528, 768, 1), 0); del buf149  # reuse
        # Source Nodes: [x_61, x_66, x_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf151, buf120, buf127, arg103_1, arg109_1, arg25_1, arg26_1, buf155, 1568, 768, grid=grid(1568), stream=stream0)
        del arg103_1
        del arg109_1
        del arg25_1
        del arg26_1
        buf156 = reinterpret_tensor(buf126, (1568, 3072), (3072, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (1568, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 3072), (1, 768), 0), out=buf156)
        del arg110_1
        buf157 = reinterpret_tensor(buf156, (8, 196, 3072), (602112, 3072, 1), 0); del buf156  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf157, arg111_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg111_1
        buf158 = reinterpret_tensor(buf155, (1568, 768), (768, 1), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg112_1, (3072, 768), (1, 3072), 0), out=buf158)
        del arg112_1
        buf162 = reinterpret_tensor(buf127, (8, 196, 768), (150528, 768, 1), 0); del buf127  # reuse
        # Source Nodes: [x_75, x_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf151, buf158, arg113_1, arg27_1, arg28_1, buf162, 1568, 768, grid=grid(1568), stream=stream0)
        del arg27_1
        del arg28_1
        buf163 = buf132; del buf132  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (1568, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 1536), (1, 768), 0), out=buf163)
        del arg114_1
        buf164 = reinterpret_tensor(buf120, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf120  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf163, buf164, 1204224, grid=grid(1204224), stream=stream0)
        buf165 = reinterpret_tensor(buf133, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf133  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf163, buf165, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf166 = reinterpret_tensor(buf146, (128, 196, 196), (38416, 196, 1), 0); del buf146  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf165, (128, 48, 196), (9408, 196, 1), 0), out=buf166)
        buf167 = reinterpret_tensor(buf142, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf142  # reuse
        buf168 = reinterpret_tensor(buf141, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf141  # reuse
        # Source Nodes: [patch_score_10, patch_score_11], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf166, buf167, buf168, 25088, 196, grid=grid(25088), stream=stream0)
        buf169 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_5, setitem_15, setitem_16, setitem_17, to_5], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf169, 115248, grid=grid(115248), stream=stream0)
        buf170 = buf139; del buf139  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf169, buf170, 921984, grid=grid(921984), stream=stream0)
        buf171 = reinterpret_tensor(buf143, (307328, 16), (16, 1), 0); del buf143  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (307328, 3), (3, 1), 0), reinterpret_tensor(arg115_1, (3, 16), (1, 3), 0), out=buf171)
        del arg115_1
        buf172 = reinterpret_tensor(buf137, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf137  # reuse
        buf173 = reinterpret_tensor(buf136, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf136  # reuse
        # Source Nodes: [pos_score_17], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf171, arg116_1, buf172, buf173, 25088, 196, grid=grid(25088), stream=stream0)
        buf174 = reinterpret_tensor(buf166, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf166  # reuse
        # Source Nodes: [attn_20, mul_16, mul_17, patch_score_10, patch_score_11, pos_score_17, sigmoid_10, sigmoid_11, sub_11], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf174, arg29_1, buf167, buf168, buf171, arg116_1, buf172, buf173, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg116_1
        del arg29_1
        buf177 = reinterpret_tensor(buf171, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf171  # reuse
        # Source Nodes: [attn_21, sum_6], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf174, buf177, 25088, 196, grid=grid(25088), stream=stream0)
        buf176 = reinterpret_tensor(buf165, (1568, 768), (768, 1), 0); del buf165  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (1568, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf176)
        del arg117_1
        buf178 = reinterpret_tensor(buf162, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf162  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf176, buf178, 1204224, grid=grid(1204224), stream=stream0)
        buf179 = reinterpret_tensor(buf176, (128, 196, 48), (9408, 48, 1), 0); del buf176  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf177, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf178, (128, 196, 48), (9408, 48, 1), 0), out=buf179)
        buf180 = reinterpret_tensor(buf178, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf178  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf179, buf180, 1204224, grid=grid(1204224), stream=stream0)
        buf181 = reinterpret_tensor(buf179, (1568, 768), (768, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (1568, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), out=buf181)
        del arg118_1
        buf182 = reinterpret_tensor(buf181, (8, 196, 768), (150528, 768, 1), 0); del buf181  # reuse
        buf186 = reinterpret_tensor(buf180, (8, 196, 768), (150528, 768, 1), 0); del buf180  # reuse
        # Source Nodes: [x_75, x_80, x_81], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf182, buf151, buf158, arg113_1, arg119_1, arg30_1, arg31_1, buf186, 1568, 768, grid=grid(1568), stream=stream0)
        del arg113_1
        del arg119_1
        del arg30_1
        del arg31_1
        buf187 = reinterpret_tensor(buf157, (1568, 3072), (3072, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (1568, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 3072), (1, 768), 0), out=buf187)
        del arg120_1
        buf188 = reinterpret_tensor(buf187, (8, 196, 3072), (602112, 3072, 1), 0); del buf187  # reuse
        # Source Nodes: [x_83], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf188, arg121_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg121_1
        buf189 = reinterpret_tensor(buf186, (1568, 768), (768, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg122_1, (3072, 768), (1, 3072), 0), out=buf189)
        del arg122_1
        buf193 = reinterpret_tensor(buf158, (8, 196, 768), (150528, 768, 1), 0); del buf158  # reuse
        # Source Nodes: [x_89, x_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf182, buf189, arg123_1, arg32_1, arg33_1, buf193, 1568, 768, grid=grid(1568), stream=stream0)
        del arg32_1
        del arg33_1
        buf194 = buf163; del buf163  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (1568, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 1536), (1, 768), 0), out=buf194)
        del arg124_1
        buf195 = reinterpret_tensor(buf151, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf151  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf194, buf195, 1204224, grid=grid(1204224), stream=stream0)
        buf196 = reinterpret_tensor(buf164, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf164  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf194, buf196, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf197 = reinterpret_tensor(buf177, (128, 196, 196), (38416, 196, 1), 0); del buf177  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf195, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf196, (128, 48, 196), (9408, 196, 1), 0), out=buf197)
        buf198 = reinterpret_tensor(buf173, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf173  # reuse
        buf199 = reinterpret_tensor(buf172, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf172  # reuse
        # Source Nodes: [patch_score_12, patch_score_13], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf197, buf198, buf199, 25088, 196, grid=grid(25088), stream=stream0)
        buf200 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_6, setitem_18, setitem_19, setitem_20, to_6], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf200, 115248, grid=grid(115248), stream=stream0)
        buf201 = buf170; del buf170  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf200, buf201, 921984, grid=grid(921984), stream=stream0)
        buf202 = reinterpret_tensor(buf174, (307328, 16), (16, 1), 0); del buf174  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (307328, 3), (3, 1), 0), reinterpret_tensor(arg125_1, (3, 16), (1, 3), 0), out=buf202)
        del arg125_1
        buf203 = reinterpret_tensor(buf168, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf168  # reuse
        buf204 = reinterpret_tensor(buf167, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf167  # reuse
        # Source Nodes: [pos_score_20], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf202, arg126_1, buf203, buf204, 25088, 196, grid=grid(25088), stream=stream0)
        buf205 = reinterpret_tensor(buf197, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf197  # reuse
        # Source Nodes: [attn_24, mul_19, mul_20, patch_score_12, patch_score_13, pos_score_20, sigmoid_12, sigmoid_13, sub_13], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf205, arg34_1, buf198, buf199, buf202, arg126_1, buf203, buf204, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg126_1
        del arg34_1
        buf208 = reinterpret_tensor(buf202, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf202  # reuse
        # Source Nodes: [attn_25, sum_7], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf205, buf208, 25088, 196, grid=grid(25088), stream=stream0)
        buf207 = reinterpret_tensor(buf196, (1568, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (1568, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 768), (1, 768), 0), out=buf207)
        del arg127_1
        buf209 = reinterpret_tensor(buf193, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf193  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf207, buf209, 1204224, grid=grid(1204224), stream=stream0)
        buf210 = reinterpret_tensor(buf207, (128, 196, 48), (9408, 48, 1), 0); del buf207  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf209, (128, 196, 48), (9408, 48, 1), 0), out=buf210)
        buf211 = reinterpret_tensor(buf209, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf209  # reuse
        # Source Nodes: [x_91], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf210, buf211, 1204224, grid=grid(1204224), stream=stream0)
        buf212 = reinterpret_tensor(buf210, (1568, 768), (768, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (1568, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), out=buf212)
        del arg128_1
        buf213 = reinterpret_tensor(buf212, (8, 196, 768), (150528, 768, 1), 0); del buf212  # reuse
        buf217 = reinterpret_tensor(buf211, (8, 196, 768), (150528, 768, 1), 0); del buf211  # reuse
        # Source Nodes: [x_89, x_94, x_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf213, buf182, buf189, arg123_1, arg129_1, arg35_1, arg36_1, buf217, 1568, 768, grid=grid(1568), stream=stream0)
        del arg123_1
        del arg129_1
        del arg35_1
        del arg36_1
        buf218 = reinterpret_tensor(buf188, (1568, 3072), (3072, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (1568, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), out=buf218)
        del arg130_1
        buf219 = reinterpret_tensor(buf218, (8, 196, 3072), (602112, 3072, 1), 0); del buf218  # reuse
        # Source Nodes: [x_97], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf219, arg131_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg131_1
        buf220 = reinterpret_tensor(buf217, (1568, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), out=buf220)
        del arg132_1
        buf224 = reinterpret_tensor(buf189, (8, 196, 768), (150528, 768, 1), 0); del buf189  # reuse
        # Source Nodes: [x_103, x_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf213, buf220, arg133_1, arg37_1, arg38_1, buf224, 1568, 768, grid=grid(1568), stream=stream0)
        del arg37_1
        del arg38_1
        buf225 = buf194; del buf194  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (1568, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 1536), (1, 768), 0), out=buf225)
        del arg134_1
        buf226 = reinterpret_tensor(buf182, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf182  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf225, buf226, 1204224, grid=grid(1204224), stream=stream0)
        buf227 = reinterpret_tensor(buf195, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf195  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf225, buf227, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf228 = reinterpret_tensor(buf208, (128, 196, 196), (38416, 196, 1), 0); del buf208  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf226, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf227, (128, 48, 196), (9408, 196, 1), 0), out=buf228)
        buf229 = reinterpret_tensor(buf204, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf204  # reuse
        buf230 = reinterpret_tensor(buf203, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf203  # reuse
        # Source Nodes: [patch_score_14, patch_score_15], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf228, buf229, buf230, 25088, 196, grid=grid(25088), stream=stream0)
        buf231 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_7, setitem_21, setitem_22, setitem_23, to_7], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf231, 115248, grid=grid(115248), stream=stream0)
        buf232 = buf201; del buf201  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf231, buf232, 921984, grid=grid(921984), stream=stream0)
        buf233 = reinterpret_tensor(buf205, (307328, 16), (16, 1), 0); del buf205  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (307328, 3), (3, 1), 0), reinterpret_tensor(arg135_1, (3, 16), (1, 3), 0), out=buf233)
        del arg135_1
        buf234 = reinterpret_tensor(buf199, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf199  # reuse
        buf235 = reinterpret_tensor(buf198, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf198  # reuse
        # Source Nodes: [pos_score_23], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf233, arg136_1, buf234, buf235, 25088, 196, grid=grid(25088), stream=stream0)
        buf236 = reinterpret_tensor(buf228, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf228  # reuse
        # Source Nodes: [attn_28, mul_22, mul_23, patch_score_14, patch_score_15, pos_score_23, sigmoid_14, sigmoid_15, sub_15], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf236, arg39_1, buf229, buf230, buf233, arg136_1, buf234, buf235, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg136_1
        del arg39_1
        buf239 = reinterpret_tensor(buf233, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf233  # reuse
        # Source Nodes: [attn_29, sum_8], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf236, buf239, 25088, 196, grid=grid(25088), stream=stream0)
        buf238 = reinterpret_tensor(buf227, (1568, 768), (768, 1), 0); del buf227  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (1568, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), out=buf238)
        del arg137_1
        buf240 = reinterpret_tensor(buf224, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf224  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf238, buf240, 1204224, grid=grid(1204224), stream=stream0)
        buf241 = reinterpret_tensor(buf238, (128, 196, 48), (9408, 48, 1), 0); del buf238  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf240, (128, 196, 48), (9408, 48, 1), 0), out=buf241)
        buf242 = reinterpret_tensor(buf240, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf240  # reuse
        # Source Nodes: [x_105], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf241, buf242, 1204224, grid=grid(1204224), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (1568, 768), (768, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), out=buf243)
        del arg138_1
        buf244 = reinterpret_tensor(buf243, (8, 196, 768), (150528, 768, 1), 0); del buf243  # reuse
        buf248 = reinterpret_tensor(buf242, (8, 196, 768), (150528, 768, 1), 0); del buf242  # reuse
        # Source Nodes: [x_103, x_108, x_109], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf244, buf213, buf220, arg133_1, arg139_1, arg40_1, arg41_1, buf248, 1568, 768, grid=grid(1568), stream=stream0)
        del arg133_1
        del arg139_1
        del arg40_1
        del arg41_1
        buf249 = reinterpret_tensor(buf219, (1568, 3072), (3072, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (1568, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 3072), (1, 768), 0), out=buf249)
        del arg140_1
        buf250 = reinterpret_tensor(buf249, (8, 196, 3072), (602112, 3072, 1), 0); del buf249  # reuse
        # Source Nodes: [x_111], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf250, arg141_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg141_1
        buf251 = reinterpret_tensor(buf248, (1568, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg142_1, (3072, 768), (1, 3072), 0), out=buf251)
        del arg142_1
        buf255 = reinterpret_tensor(buf220, (8, 196, 768), (150528, 768, 1), 0); del buf220  # reuse
        # Source Nodes: [x_117, x_118], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf244, buf251, arg143_1, arg42_1, arg43_1, buf255, 1568, 768, grid=grid(1568), stream=stream0)
        del arg42_1
        del arg43_1
        buf256 = buf225; del buf225  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (1568, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 1536), (1, 768), 0), out=buf256)
        del arg144_1
        buf257 = reinterpret_tensor(buf213, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf213  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf256, buf257, 1204224, grid=grid(1204224), stream=stream0)
        buf258 = reinterpret_tensor(buf226, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf226  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf256, buf258, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf259 = reinterpret_tensor(buf239, (128, 196, 196), (38416, 196, 1), 0); del buf239  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf258, (128, 48, 196), (9408, 196, 1), 0), out=buf259)
        buf260 = reinterpret_tensor(buf235, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf235  # reuse
        buf261 = reinterpret_tensor(buf234, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf234  # reuse
        # Source Nodes: [patch_score_16, patch_score_17], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf259, buf260, buf261, 25088, 196, grid=grid(25088), stream=stream0)
        buf262 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_8, setitem_24, setitem_25, setitem_26, to_8], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf262, 115248, grid=grid(115248), stream=stream0)
        buf263 = buf232; del buf232  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf262, buf263, 921984, grid=grid(921984), stream=stream0)
        buf264 = reinterpret_tensor(buf236, (307328, 16), (16, 1), 0); del buf236  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (307328, 3), (3, 1), 0), reinterpret_tensor(arg145_1, (3, 16), (1, 3), 0), out=buf264)
        del arg145_1
        buf265 = reinterpret_tensor(buf230, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf230  # reuse
        buf266 = reinterpret_tensor(buf229, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf229  # reuse
        # Source Nodes: [pos_score_26], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf264, arg146_1, buf265, buf266, 25088, 196, grid=grid(25088), stream=stream0)
        buf267 = reinterpret_tensor(buf259, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf259  # reuse
        # Source Nodes: [attn_32, mul_25, mul_26, patch_score_16, patch_score_17, pos_score_26, sigmoid_16, sigmoid_17, sub_17], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf267, arg44_1, buf260, buf261, buf264, arg146_1, buf265, buf266, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg146_1
        del arg44_1
        buf270 = reinterpret_tensor(buf264, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf264  # reuse
        # Source Nodes: [attn_33, sum_9], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf267, buf270, 25088, 196, grid=grid(25088), stream=stream0)
        buf269 = reinterpret_tensor(buf258, (1568, 768), (768, 1), 0); del buf258  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (1568, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 768), (1, 768), 0), out=buf269)
        del arg147_1
        buf271 = reinterpret_tensor(buf255, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf255  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf269, buf271, 1204224, grid=grid(1204224), stream=stream0)
        buf272 = reinterpret_tensor(buf269, (128, 196, 48), (9408, 48, 1), 0); del buf269  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf271, (128, 196, 48), (9408, 48, 1), 0), out=buf272)
        buf273 = reinterpret_tensor(buf271, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf271  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf272, buf273, 1204224, grid=grid(1204224), stream=stream0)
        buf274 = reinterpret_tensor(buf272, (1568, 768), (768, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (1568, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 768), (1, 768), 0), out=buf274)
        del arg148_1
        buf275 = reinterpret_tensor(buf274, (8, 196, 768), (150528, 768, 1), 0); del buf274  # reuse
        buf279 = reinterpret_tensor(buf273, (8, 196, 768), (150528, 768, 1), 0); del buf273  # reuse
        # Source Nodes: [x_117, x_122, x_123], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf275, buf244, buf251, arg143_1, arg149_1, arg45_1, arg46_1, buf279, 1568, 768, grid=grid(1568), stream=stream0)
        del arg143_1
        del arg149_1
        del arg45_1
        del arg46_1
        buf280 = reinterpret_tensor(buf250, (1568, 3072), (3072, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (1568, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 3072), (1, 768), 0), out=buf280)
        del arg150_1
        buf281 = reinterpret_tensor(buf280, (8, 196, 3072), (602112, 3072, 1), 0); del buf280  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf281, arg151_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg151_1
        buf282 = reinterpret_tensor(buf279, (1568, 768), (768, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg152_1, (3072, 768), (1, 3072), 0), out=buf282)
        del arg152_1
        buf286 = reinterpret_tensor(buf251, (8, 196, 768), (150528, 768, 1), 0); del buf251  # reuse
        # Source Nodes: [x_131, x_132], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf275, buf282, arg153_1, arg47_1, arg48_1, buf286, 1568, 768, grid=grid(1568), stream=stream0)
        del arg47_1
        del arg48_1
        buf287 = buf256; del buf256  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1568, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 1536), (1, 768), 0), out=buf287)
        del arg154_1
        buf288 = reinterpret_tensor(buf244, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf244  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf287, buf288, 1204224, grid=grid(1204224), stream=stream0)
        buf289 = reinterpret_tensor(buf257, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf257  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf287, buf289, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del buf287
        buf290 = reinterpret_tensor(buf270, (128, 196, 196), (38416, 196, 1), 0); del buf270  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf288, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf289, (128, 48, 196), (9408, 196, 1), 0), out=buf290)
        del buf288
        buf291 = reinterpret_tensor(buf266, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf266  # reuse
        buf292 = reinterpret_tensor(buf265, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf265  # reuse
        # Source Nodes: [patch_score_18, patch_score_19], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_5.run(buf290, buf291, buf292, 25088, 196, grid=grid(25088), stream=stream0)
        buf293 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices_9, setitem_27, setitem_28, setitem_29, to_9], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_6.run(buf293, 115248, grid=grid(115248), stream=stream0)
        buf294 = buf263; del buf263  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_pos_proj], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf293, buf294, 921984, grid=grid(921984), stream=stream0)
        buf295 = reinterpret_tensor(buf267, (307328, 16), (16, 1), 0); del buf267  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf294, (307328, 3), (3, 1), 0), reinterpret_tensor(arg155_1, (3, 16), (1, 3), 0), out=buf295)
        del arg155_1
        del buf294
        buf296 = reinterpret_tensor(buf261, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf261  # reuse
        buf297 = reinterpret_tensor(buf260, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf260  # reuse
        # Source Nodes: [pos_score_29], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf295, arg156_1, buf296, buf297, 25088, 196, grid=grid(25088), stream=stream0)
        buf298 = reinterpret_tensor(buf290, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf290  # reuse
        # Source Nodes: [attn_36, mul_28, mul_29, patch_score_18, patch_score_19, pos_score_29, sigmoid_18, sigmoid_19, sub_19], Original ATen: [aten._softmax, aten.add, aten.mul, aten.rsub, aten.sigmoid]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_9.run(buf298, arg49_1, buf291, buf292, buf295, arg156_1, buf296, buf297, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg156_1
        del arg49_1
        del buf291
        del buf292
        del buf296
        del buf297
        buf301 = reinterpret_tensor(buf295, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf295  # reuse
        # Source Nodes: [attn_37, sum_10], Original ATen: [aten.div, aten.sum]
        triton_per_fused_div_sum_10.run(buf298, buf301, 25088, 196, grid=grid(25088), stream=stream0)
        del buf298
        buf300 = reinterpret_tensor(buf289, (1568, 768), (768, 1), 0); del buf289  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1568, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), out=buf300)
        del arg157_1
        buf302 = reinterpret_tensor(buf286, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf286  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf300, buf302, 1204224, grid=grid(1204224), stream=stream0)
        buf303 = reinterpret_tensor(buf300, (128, 196, 48), (9408, 48, 1), 0); del buf300  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf301, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf302, (128, 196, 48), (9408, 48, 1), 0), out=buf303)
        del buf301
        buf304 = reinterpret_tensor(buf302, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf302  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf303, buf304, 1204224, grid=grid(1204224), stream=stream0)
        buf305 = reinterpret_tensor(buf303, (1568, 768), (768, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (1568, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), out=buf305)
        del arg158_1
        buf306 = reinterpret_tensor(buf305, (8, 196, 768), (150528, 768, 1), 0); del buf305  # reuse
        buf310 = reinterpret_tensor(buf304, (8, 196, 768), (150528, 768, 1), 0); del buf304  # reuse
        # Source Nodes: [x_131, x_136, x_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf306, buf275, buf282, arg153_1, arg159_1, arg50_1, arg51_1, buf310, 1568, 768, grid=grid(1568), stream=stream0)
        del arg153_1
        del arg159_1
        del arg50_1
        del arg51_1
        del buf275
        del buf282
        buf311 = reinterpret_tensor(buf281, (1568, 3072), (3072, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 3072), (1, 768), 0), out=buf311)
        del arg160_1
        buf312 = reinterpret_tensor(buf311, (8, 196, 3072), (602112, 3072, 1), 0); del buf311  # reuse
        # Source Nodes: [x_139], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf312, arg161_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg161_1
        buf313 = reinterpret_tensor(buf310, (1568, 768), (768, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg162_1, (3072, 768), (1, 3072), 0), out=buf313)
        del arg162_1
        del buf312
        buf317 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, x_147], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_17.run(arg1_1, buf306, buf313, arg163_1, arg52_1, arg53_1, buf317, 1576, 768, grid=grid(1576), stream=stream0)
        del arg52_1
        del arg53_1
        buf318 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (1576, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 2304), (1, 768), 0), out=buf318)
        del arg164_1
        buf319 = reinterpret_tensor(buf317, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf317  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf318, buf319, 1210368, grid=grid(1210368), stream=stream0)
        buf320 = empty((8, 16, 48, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf318, buf320, 6144, 197, grid=grid(6144, 197), stream=stream0)
        buf321 = empty((128, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf319, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf320, (128, 48, 197), (9456, 197, 1), 0), out=buf321)
        buf324 = empty((8, 16, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_40, attn_41], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_20.run(buf321, buf324, 25216, 197, grid=grid(25216), stream=stream0)
        buf325 = reinterpret_tensor(buf320, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf320  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf318, buf325, 1210368, grid=grid(1210368), stream=stream0)
        buf326 = reinterpret_tensor(buf319, (128, 197, 48), (9456, 48, 1), 0); del buf319  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf325, (128, 197, 48), (9456, 48, 1), 0), out=buf326)
        buf327 = reinterpret_tensor(buf325, (8, 197, 16, 48), (151296, 768, 48, 1), 0); del buf325  # reuse
        # Source Nodes: [x_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf326, buf327, 1210368, grid=grid(1210368), stream=stream0)
        buf328 = reinterpret_tensor(buf326, (1576, 768), (768, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf327, (1576, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 768), (1, 768), 0), out=buf328)
        del arg165_1
        buf329 = reinterpret_tensor(buf328, (8, 197, 768), (151296, 768, 1), 0); del buf328  # reuse
        buf333 = reinterpret_tensor(buf327, (8, 197, 768), (151296, 768, 1), 0); del buf327  # reuse
        # Source Nodes: [cat_1, x_151, x_152], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_23.run(buf329, arg1_1, buf306, buf313, arg163_1, arg166_1, arg54_1, arg55_1, buf333, 1576, 768, grid=grid(1576), stream=stream0)
        del arg163_1
        del arg166_1
        del arg1_1
        del arg54_1
        del arg55_1
        del buf306
        del buf313
        buf334 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf333, (1576, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 3072), (1, 768), 0), out=buf334)
        del arg167_1
        buf335 = reinterpret_tensor(buf334, (8, 197, 3072), (605184, 3072, 1), 0); del buf334  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_24.run(buf335, arg168_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg168_1
        buf336 = reinterpret_tensor(buf333, (1576, 768), (768, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg169_1, (3072, 768), (1, 3072), 0), out=buf336)
        del arg169_1
        buf340 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160, x_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf329, buf336, arg170_1, arg56_1, arg57_1, buf340, 1576, 768, grid=grid(1576), stream=stream0)
        del arg56_1
        del arg57_1
        buf341 = buf318; del buf318  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (1576, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 2304), (1, 768), 0), out=buf341)
        del arg171_1
        buf342 = reinterpret_tensor(buf340, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf340  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf341, buf342, 1210368, grid=grid(1210368), stream=stream0)
        buf343 = empty((8, 16, 48, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf341, buf343, 6144, 197, grid=grid(6144, 197), stream=stream0)
        buf344 = reinterpret_tensor(buf324, (128, 197, 197), (38809, 197, 1), 0); del buf324  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf342, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf343, (128, 48, 197), (9456, 197, 1), 0), out=buf344)
        buf347 = reinterpret_tensor(buf321, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf321  # reuse
        # Source Nodes: [attn_43, attn_44], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_20.run(buf344, buf347, 25216, 197, grid=grid(25216), stream=stream0)
        del buf344
        buf348 = reinterpret_tensor(buf343, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf343  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf341, buf348, 1210368, grid=grid(1210368), stream=stream0)
        del buf341
        buf349 = reinterpret_tensor(buf342, (128, 197, 48), (9456, 48, 1), 0); del buf342  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf347, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf348, (128, 197, 48), (9456, 48, 1), 0), out=buf349)
        del buf347
        buf350 = reinterpret_tensor(buf348, (8, 197, 16, 48), (151296, 768, 48, 1), 0); del buf348  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf349, buf350, 1210368, grid=grid(1210368), stream=stream0)
        buf351 = reinterpret_tensor(buf349, (1576, 768), (768, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (1576, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), out=buf351)
        del arg172_1
        buf352 = reinterpret_tensor(buf351, (8, 197, 768), (151296, 768, 1), 0); del buf351  # reuse
        buf356 = reinterpret_tensor(buf350, (8, 197, 768), (151296, 768, 1), 0); del buf350  # reuse
        # Source Nodes: [x_160, x_165, x_166], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf352, buf329, buf336, arg170_1, arg173_1, arg58_1, arg59_1, buf356, 1576, 768, grid=grid(1576), stream=stream0)
        del arg170_1
        del arg173_1
        del arg58_1
        del arg59_1
        del buf329
        del buf336
        buf357 = reinterpret_tensor(buf335, (1576, 3072), (3072, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (1576, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 3072), (1, 768), 0), out=buf357)
        del arg174_1
        buf358 = reinterpret_tensor(buf357, (8, 197, 3072), (605184, 3072, 1), 0); del buf357  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_24.run(buf358, arg175_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg175_1
        buf359 = reinterpret_tensor(buf356, (1576, 768), (768, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg176_1, (3072, 768), (1, 3072), 0), out=buf359)
        del arg176_1
        del buf358
        buf360 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf361 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174, x_177], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf352, buf359, arg177_1, buf360, buf361, 1576, 768, grid=grid(1576), stream=stream0)
        buf363 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf352, buf359, arg177_1, buf360, buf361, arg60_1, arg61_1, buf363, 6144, grid=grid(6144), stream=stream0)
        del arg177_1
        del arg60_1
        del arg61_1
        del buf352
        del buf359
        del buf360
        del buf361
        buf364 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179, x_180], Original ATen: [aten.addmm, aten.clone]
        extern_kernels.addmm(arg179_1, buf363, reinterpret_tensor(arg178_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf364)
        del arg178_1
        del arg179_1
        return (buf364, buf14, buf45, buf76, buf107, buf138, buf169, buf200, buf231, buf262, buf293, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convit_base', benchmark_compiled_module)
