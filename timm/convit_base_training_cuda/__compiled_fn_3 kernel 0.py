
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


# kernel path: /tmp/torchinductor_youkaichao/3e/c3ez6hvcby6osokhfy6rknpeykwrnp27bm2ctmy3bhoa533kkzgs.py
# Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_4 => add
# x_6 => add_1, clone_1, rsqrt, var_mean
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': []}
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
    tmp16 = 768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtfyewwizisdbarsbcfbzpmmixdvvz4gtn5va4od57fepptmqzg.py
# Source Nodes: [l__mod___blocks_0_attn_qk, x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# l__mod___blocks_0_attn_qk => view_5
# x_4 => add
# x_6 => add_1, add_2, clone_1, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_add_native_layer_norm_view_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_2', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp13, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (768*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5budphlbg75qf4udge5qsvx5madcpldoan7ygij2wyofvqnf5ba.py
# Source Nodes: [rel_indices, setitem, setitem_1, setitem_2, to], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
# rel_indices => full
# setitem => copy, select_scatter
# setitem_1 => copy_1, select_scatter_1
# setitem_2 => copy_2, select_scatter_2
# to => device_put
triton_poi_fused__to_copy_copy_select_scatter_zeros_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_select_scatter_zeros_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4rsvyzxjrgrjihtx5k5ojtgmi6c52ht6vmk2sgthdq3ysf3zgt.py
# Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten._unsafe_view, aten.clone]
# l__mod___blocks_0_attn_pos_proj => clone_4, view_8
triton_poi_fused__unsafe_view_clone_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3*(x1 % 38416))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygt5q3jryvqacqskkesx5w6s7pjqn42porvc7mvca2ve2dey256.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_5
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmp5fzasmey5jjwn7ymohb2643mmmt5llld7btykidjcfxqvxkv.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_6
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4u/c4udiag6v3r5tuxq7oebqm2edotfmq6oqjnyytwlr6ddxqtc2gwi.py
# Source Nodes: [pos_score_2], Original ATen: [aten._softmax]
# pos_score_2 => amax_1, clone_7, div_1, exp_1, sub_3, sum_2
triton_red_fused__softmax_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + (x0 + (16*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 + tmp1
        tmp8 = tmp7 - tmp4
        tmp9 = tl.exp(tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    x4 = (xindex // 16) % 196
    x5 = (xindex // 3136)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp13 = tl.load(in_ptr0 + (x0 + (16*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp13 + tmp1
        tmp15 = tmp14 - tmp4
        tmp16 = tl.exp(tmp15)
        tmp17 = tmp16 / tmp11
        tl.store(out_ptr2 + (r2 + (196*x4) + (38416*x0) + (614656*x5)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvvtcks3uiqy56dduc2oehq6se2mog6x7g5pdq4xlaa5tb2hnnq.py
# Source Nodes: [attn, attn_1, mul_1, mul_2, patch_score, patch_score_1, sigmoid, sub_1, sum_1], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
# attn => add_5
# attn_1 => div_2
# mul_1 => mul_3
# mul_2 => mul_4
# patch_score => mul_2
# patch_score_1 => amax, div, exp, sub_2, sum_1
# sigmoid => sigmoid
# sub_1 => sub_4
# sum_1 => sum_3
triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x3 = (xindex // 196) % 16
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask & xmask, other=0.0)
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = 1.0
    tmp17 = tmp16 - tmp15
    tmp18 = tmp17 * tmp13
    tmp20 = tmp15 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp21 / tmp25
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (196*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdfphg5uw3jdqhidzbliht7zui64q4jdlwg4bxfri4b2ufjyvb6h.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_10
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rt/crthjs5db4rpafpkimgrnlm5uzh4gxif2yl66wl5yxyfzumvrvhd.py
# Source Nodes: [x_8], Original ATen: [aten.view]
# x_8 => view_21
triton_poi_fused_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + ((48*(x1 % 196)) + (9408*(x0 // 48)) + (150528*(x1 // 196)) + (x0 % 48)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahikkcqjk5hvnkfb6zj5vpq2hkxcc2dotomreshf7lxc4sub6gc.py
# Source Nodes: [x_10, x_11, x_12, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_10 => add_6
# x_11 => add_7, add_8, clone_13, mul_5, mul_6, rsqrt_1, sub_5, var_mean_1
# x_12 => view_23
# x_4 => add
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmgjdutq4yfrnvzgylofiinlpvw3uidvtt7hgi3zv5qiegcjwjt.py
# Source Nodes: [x_13, x_16], Original ATen: [aten.gelu, aten.view]
# x_13 => add_9, erf, mul_7, mul_8, mul_9
# x_16 => view_25
triton_poi_fused_gelu_view_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_12', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7l53frzv55nuxdkieky4odkafvgj6vsjoya3jvnhd3udaz44n5.py
# Source Nodes: [l__mod___blocks_1_attn_qk, x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_attn_qk => view_31
# x_19 => add_10
# x_20 => add_11, add_12, clone_16, mul_10, mul_11, rsqrt_2, sub_6, var_mean_2
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphdlajcnv3ucvs7fguhxfpuk7wsk52x2wd3dwauvxyhycrqywka.py
# Source Nodes: [x_19, x_24, x_25, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_19 => add_10
# x_24 => add_16
# x_25 => add_17, add_18, clone_28, mul_15, mul_16, rsqrt_3, sub_11, var_mean_3
# x_26 => view_49
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2rlth3d3wsoru5pi2p6bb33f2lnnw6ptsipvl5bjrsfjutl2if.py
# Source Nodes: [cat_1, l__mod___blocks_10_attn_qkv, x_147], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_1 => cat
# l__mod___blocks_10_attn_qkv => view_261
# x_147 => add_101, add_102, mul_100, mul_101, rsqrt_20, sub_60, var_mean_20
triton_per_fused_cat_native_layer_norm_view_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp35 = 768.0
    tmp36 = tmp34 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp18 - tmp28
    tmp41 = tmp40 * tmp39
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp39, xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkg7xe3mvapvozsu4lyweaf5ovaumahmuqedcan5falhla45ban.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_151
triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zn/cznpfzh6dlymcwigjwgudrrhj7dzee4xxmf6xzzpuj5oral6dufc.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_152
triton_poi_fused_clone_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2l/c2labudhs5if23ssfsqgfwuechz5vkfvr27tubzecsnjcqlf3tzh.py
# Source Nodes: [attn_40, attn_41, attn_42], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
# attn_40 => mul_102
# attn_41 => amax_20, div_30, exp_20, sub_61, sum_31
# attn_42 => clone_153
triton_per_fused__softmax_clone_detach_mul_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_mul_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr3 + (r1 + (197*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufjm5yeuiysuiseexntnzc4ei4336ukrvw2yhuto3eycpat2cet.py
# Source Nodes: [matmul_21], Original ATen: [aten.clone]
# matmul_21 => clone_154
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2r/c2rfsfnmgrloggs65vgkvr42o7jfn5cpzq34x5omdnrbfmf5kagc.py
# Source Nodes: [x_149], Original ATen: [aten.view]
# x_149 => view_271
triton_poi_fused_view_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((48*(x1 % 197)) + (9456*(x0 // 48)) + (151296*(x1 // 197)) + (x0 % 48)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce64fhdsa6chikj3vs7ufvzhuy37zb7g4owvrglndriztq37jpwi.py
# Source Nodes: [x_151, x_152, x_153], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_151 => add_103
# x_152 => add_104, add_105, mul_103, mul_104, rsqrt_21, sub_62, var_mean_21
# x_153 => view_273
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cux5artznmohxqxlc3uqgnymanmnsvgw7xjuugy4fiab3kjeeyii.py
# Source Nodes: [x_154, x_157], Original ATen: [aten.gelu, aten.view]
# x_154 => add_106, erf_10, mul_105, mul_106, mul_107
# x_157 => view_275
triton_poi_fused_gelu_view_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4841472
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


# kernel path: /tmp/torchinductor_youkaichao/fi/cfita52siiogj3pzmmok577byksybaufb6dfhctmy3q32mjnit2i.py
# Source Nodes: [l__mod___blocks_11_attn_qkv, x_151, x_160, x_161], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_11_attn_qkv => view_277
# x_151 => add_103
# x_160 => add_107
# x_161 => add_108, add_109, mul_108, mul_109, rsqrt_22, sub_63, var_mean_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vhbnjwkiqgtolm3q3oxpso2uxmo32t53cvdjelbv3yyzfy6c2g.py
# Source Nodes: [x_165, x_174, x_177], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_165 => add_110
# x_174 => add_114
# x_177 => add_115, mul_116, rsqrt_24, sub_66, var_mean_24
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp32 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmm7zx3uss7peolqb2qq674qhv4irvncmg5c3j5ooaozzbdudtoh.py
# Source Nodes: [x_179], Original ATen: [aten.clone]
# x_179 => clone_167
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (151296*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181 = args
    args.clear()
    assert_size_stride(primals_1, (1, 196, 768), (150528, 768, 1))
    assert_size_stride(primals_2, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (16, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (16, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (1536, 768), (768, 1))
    assert_size_stride(primals_66, (16, 3), (3, 1))
    assert_size_stride(primals_67, (16, ), (1, ))
    assert_size_stride(primals_68, (768, 768), (768, 1))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (3072, 768), (768, 1))
    assert_size_stride(primals_72, (3072, ), (1, ))
    assert_size_stride(primals_73, (768, 3072), (3072, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (1536, 768), (768, 1))
    assert_size_stride(primals_76, (16, 3), (3, 1))
    assert_size_stride(primals_77, (16, ), (1, ))
    assert_size_stride(primals_78, (768, 768), (768, 1))
    assert_size_stride(primals_79, (768, 768), (768, 1))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (3072, 768), (768, 1))
    assert_size_stride(primals_82, (3072, ), (1, ))
    assert_size_stride(primals_83, (768, 3072), (3072, 1))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (1536, 768), (768, 1))
    assert_size_stride(primals_86, (16, 3), (3, 1))
    assert_size_stride(primals_87, (16, ), (1, ))
    assert_size_stride(primals_88, (768, 768), (768, 1))
    assert_size_stride(primals_89, (768, 768), (768, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (3072, 768), (768, 1))
    assert_size_stride(primals_92, (3072, ), (1, ))
    assert_size_stride(primals_93, (768, 3072), (3072, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (1536, 768), (768, 1))
    assert_size_stride(primals_96, (16, 3), (3, 1))
    assert_size_stride(primals_97, (16, ), (1, ))
    assert_size_stride(primals_98, (768, 768), (768, 1))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (3072, 768), (768, 1))
    assert_size_stride(primals_102, (3072, ), (1, ))
    assert_size_stride(primals_103, (768, 3072), (3072, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (1536, 768), (768, 1))
    assert_size_stride(primals_106, (16, 3), (3, 1))
    assert_size_stride(primals_107, (16, ), (1, ))
    assert_size_stride(primals_108, (768, 768), (768, 1))
    assert_size_stride(primals_109, (768, 768), (768, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (3072, 768), (768, 1))
    assert_size_stride(primals_112, (3072, ), (1, ))
    assert_size_stride(primals_113, (768, 3072), (3072, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (1536, 768), (768, 1))
    assert_size_stride(primals_116, (16, 3), (3, 1))
    assert_size_stride(primals_117, (16, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, 768), (768, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (3072, 768), (768, 1))
    assert_size_stride(primals_122, (3072, ), (1, ))
    assert_size_stride(primals_123, (768, 3072), (3072, 1))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (1536, 768), (768, 1))
    assert_size_stride(primals_126, (16, 3), (3, 1))
    assert_size_stride(primals_127, (16, ), (1, ))
    assert_size_stride(primals_128, (768, 768), (768, 1))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (3072, 768), (768, 1))
    assert_size_stride(primals_132, (3072, ), (1, ))
    assert_size_stride(primals_133, (768, 3072), (3072, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (1536, 768), (768, 1))
    assert_size_stride(primals_136, (16, 3), (3, 1))
    assert_size_stride(primals_137, (16, ), (1, ))
    assert_size_stride(primals_138, (768, 768), (768, 1))
    assert_size_stride(primals_139, (768, 768), (768, 1))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (3072, 768), (768, 1))
    assert_size_stride(primals_142, (3072, ), (1, ))
    assert_size_stride(primals_143, (768, 3072), (3072, 1))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (1536, 768), (768, 1))
    assert_size_stride(primals_146, (16, 3), (3, 1))
    assert_size_stride(primals_147, (16, ), (1, ))
    assert_size_stride(primals_148, (768, 768), (768, 1))
    assert_size_stride(primals_149, (768, 768), (768, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (3072, 768), (768, 1))
    assert_size_stride(primals_152, (3072, ), (1, ))
    assert_size_stride(primals_153, (768, 3072), (3072, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (1536, 768), (768, 1))
    assert_size_stride(primals_156, (16, 3), (3, 1))
    assert_size_stride(primals_157, (16, ), (1, ))
    assert_size_stride(primals_158, (768, 768), (768, 1))
    assert_size_stride(primals_159, (768, 768), (768, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (3072, 768), (768, 1))
    assert_size_stride(primals_162, (3072, ), (1, ))
    assert_size_stride(primals_163, (768, 3072), (3072, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (2304, 768), (768, 1))
    assert_size_stride(primals_166, (768, 768), (768, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (3072, 768), (768, 1))
    assert_size_stride(primals_169, (3072, ), (1, ))
    assert_size_stride(primals_170, (768, 3072), (3072, 1))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (2304, 768), (768, 1))
    assert_size_stride(primals_173, (768, 768), (768, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (3072, 768), (768, 1))
    assert_size_stride(primals_176, (3072, ), (1, ))
    assert_size_stride(primals_177, (768, 3072), (3072, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (1000, 768), (768, 1))
    assert_size_stride(primals_180, (1000, ), (1, ))
    assert_size_stride(primals_181, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_181, primals_63, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        buf1 = empty_strided((8, 196, 1, 6), (1176, 6, 9408, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 6), (1176, 6, 9408, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 6), (1176, 6, 9408, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_native_layer_norm_0.run(buf0, primals_64, primals_1, buf1, buf2, buf3, 9408, 128, grid=grid(9408), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf408 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_1.run(buf1, buf2, buf3, buf4, buf5, buf408, 1568, 6, grid=grid(1568), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf9 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_qk, x_4, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_2.run(buf0, primals_64, primals_1, buf4, buf5, primals_3, primals_4, buf7, buf9, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del primals_4
        buf8 = empty((1, 196, 196, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [rel_indices, setitem, setitem_1, setitem_2, to], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter, aten.zeros]
        triton_poi_fused__to_copy_copy_select_scatter_zeros_3.run(buf8, 115248, grid=grid(115248), stream=stream0)
        buf10 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_65, (768, 1536), (1, 768), 0), out=buf10)
        buf11 = empty((307328, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf8, buf11, 921984, grid=grid(921984), stream=stream0)
        buf12 = empty((307328, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_66, (3, 16), (1, 3), 0), out=buf12)
        del primals_66
        buf13 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf10, buf13, 1204224, grid=grid(1204224), stream=stream0)
        buf14 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf10, buf14, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf15 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf14, (128, 48, 196), (9408, 196, 1), 0), out=buf15)
        buf21 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf12, primals_67, buf21, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_67
        buf18 = reinterpret_tensor(buf12, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf12  # reuse
        buf22 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf24 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, mul_1, mul_2, patch_score, patch_score_1, sigmoid, sub_1, sum_1], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf15, primals_5, buf21, buf18, buf22, buf24, 25088, 196, grid=grid(25088), stream=stream0)
        buf23 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, reinterpret_tensor(primals_68, (768, 768), (1, 768), 0), out=buf23)
        buf25 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf23, buf25, 1204224, grid=grid(1204224), stream=stream0)
        buf26 = reinterpret_tensor(buf23, (128, 196, 48), (9408, 48, 1), 0); del buf23  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf25, (128, 196, 48), (9408, 48, 1), 0), out=buf26)
        buf27 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf26, buf27, 1204224, grid=grid(1204224), stream=stream0)
        buf28 = reinterpret_tensor(buf26, (1568, 768), (768, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf27, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), out=buf28)
        buf29 = reinterpret_tensor(buf28, (8, 196, 768), (150528, 768, 1), 0); del buf28  # reuse
        buf33 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf34 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf407 = reinterpret_tensor(buf5, (8, 196, 1), (196, 1, 1), 0); del buf5  # reuse
        # Source Nodes: [x_10, x_11, x_12, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf29, buf0, primals_64, primals_1, primals_70, primals_6, primals_7, buf33, buf34, buf407, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_1
        del primals_64
        del primals_7
        del primals_70
        buf35 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_72, buf34, reinterpret_tensor(primals_71, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf35)
        del primals_72
        buf36 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_16], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf35, buf36, 4816896, grid=grid(4816896), stream=stream0)
        buf37 = reinterpret_tensor(buf0, (1568, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf36, reinterpret_tensor(primals_73, (3072, 768), (1, 3072), 0), out=buf37)
        buf41 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf42 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf406 = reinterpret_tensor(buf4, (8, 196, 1), (196, 1, 1), 0); del buf4  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_qk, x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf29, buf37, primals_74, primals_8, primals_9, buf41, buf42, buf406, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_9
        buf43 = buf10; del buf10  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf42, reinterpret_tensor(primals_75, (768, 1536), (1, 768), 0), out=buf43)
        buf44 = reinterpret_tensor(buf15, (307328, 16), (16, 1), 0); del buf15  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_76, (3, 16), (1, 3), 0), out=buf44)
        del primals_76
        buf45 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf43, buf45, 1204224, grid=grid(1204224), stream=stream0)
        buf46 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf43, buf46, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf47 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf45, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf46, (128, 48, 196), (9408, 196, 1), 0), out=buf47)
        buf53 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf44, primals_77, buf53, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_77
        buf50 = reinterpret_tensor(buf44, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf44  # reuse
        buf54 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf56 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_4, attn_5, mul_4, mul_5, patch_score_2, patch_score_3, sigmoid_2, sub_3, sum_2], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf47, primals_10, buf53, buf50, buf54, buf56, 25088, 196, grid=grid(25088), stream=stream0)
        buf55 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf42, reinterpret_tensor(primals_78, (768, 768), (1, 768), 0), out=buf55)
        buf57 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf55, buf57, 1204224, grid=grid(1204224), stream=stream0)
        buf58 = reinterpret_tensor(buf55, (128, 196, 48), (9408, 48, 1), 0); del buf55  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf56, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf57, (128, 196, 48), (9408, 48, 1), 0), out=buf58)
        buf59 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf58, buf59, 1204224, grid=grid(1204224), stream=stream0)
        buf60 = reinterpret_tensor(buf58, (1568, 768), (768, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf59, reinterpret_tensor(primals_79, (768, 768), (1, 768), 0), out=buf60)
        buf61 = reinterpret_tensor(buf60, (8, 196, 768), (150528, 768, 1), 0); del buf60  # reuse
        buf65 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf66 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf405 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19, x_24, x_25, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf61, buf29, buf37, primals_74, primals_80, primals_11, primals_12, buf65, buf66, buf405, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_12
        del primals_74
        del primals_80
        buf67 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_82, buf66, reinterpret_tensor(primals_81, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf67)
        del primals_82
        buf68 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27, x_30], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf67, buf68, 4816896, grid=grid(4816896), stream=stream0)
        buf69 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf68, reinterpret_tensor(primals_83, (3072, 768), (1, 3072), 0), out=buf69)
        buf73 = buf29; del buf29  # reuse
        buf74 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf404 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_attn_qk, x_33, x_34], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf61, buf69, primals_84, primals_13, primals_14, buf73, buf74, buf404, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_14
        buf75 = buf43; del buf43  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_85, (768, 1536), (1, 768), 0), out=buf75)
        buf76 = reinterpret_tensor(buf47, (307328, 16), (16, 1), 0); del buf47  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_86, (3, 16), (1, 3), 0), out=buf76)
        del primals_86
        buf77 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf75, buf77, 1204224, grid=grid(1204224), stream=stream0)
        buf78 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf75, buf78, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf79 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf78, (128, 48, 196), (9408, 196, 1), 0), out=buf79)
        buf85 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf76, primals_87, buf85, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_87
        buf82 = reinterpret_tensor(buf76, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf76  # reuse
        buf86 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf88 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_8, attn_9, mul_7, mul_8, patch_score_4, patch_score_5, sigmoid_4, sub_5, sum_3], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf79, primals_15, buf85, buf82, buf86, buf88, 25088, 196, grid=grid(25088), stream=stream0)
        buf87 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), out=buf87)
        buf89 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf87, buf89, 1204224, grid=grid(1204224), stream=stream0)
        buf90 = reinterpret_tensor(buf87, (128, 196, 48), (9408, 48, 1), 0); del buf87  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf89, (128, 196, 48), (9408, 48, 1), 0), out=buf90)
        buf91 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf90, buf91, 1204224, grid=grid(1204224), stream=stream0)
        buf92 = reinterpret_tensor(buf90, (1568, 768), (768, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf91, reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), out=buf92)
        buf93 = reinterpret_tensor(buf92, (8, 196, 768), (150528, 768, 1), 0); del buf92  # reuse
        buf97 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf98 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf403 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_38, x_39, x_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf93, buf61, buf69, primals_84, primals_90, primals_16, primals_17, buf97, buf98, buf403, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_17
        del primals_84
        del primals_90
        buf99 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_92, buf98, reinterpret_tensor(primals_91, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf99)
        del primals_92
        buf100 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41, x_44], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf99, buf100, 4816896, grid=grid(4816896), stream=stream0)
        buf101 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf100, reinterpret_tensor(primals_93, (3072, 768), (1, 3072), 0), out=buf101)
        buf105 = buf61; del buf61  # reuse
        buf106 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf402 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_attn_qk, x_47, x_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf93, buf101, primals_94, primals_18, primals_19, buf105, buf106, buf402, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_19
        buf107 = buf75; del buf75  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_95, (768, 1536), (1, 768), 0), out=buf107)
        buf108 = reinterpret_tensor(buf79, (307328, 16), (16, 1), 0); del buf79  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_96, (3, 16), (1, 3), 0), out=buf108)
        del primals_96
        buf109 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf107, buf109, 1204224, grid=grid(1204224), stream=stream0)
        buf110 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf107, buf110, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf111 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf110, (128, 48, 196), (9408, 196, 1), 0), out=buf111)
        buf117 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_11], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf108, primals_97, buf117, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_97
        buf114 = reinterpret_tensor(buf108, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf108  # reuse
        buf118 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf120 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13, mul_10, mul_11, patch_score_6, patch_score_7, sigmoid_6, sub_7, sum_4], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf111, primals_20, buf117, buf114, buf118, buf120, 25088, 196, grid=grid(25088), stream=stream0)
        buf119 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(primals_98, (768, 768), (1, 768), 0), out=buf119)
        buf121 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf119, buf121, 1204224, grid=grid(1204224), stream=stream0)
        buf122 = reinterpret_tensor(buf119, (128, 196, 48), (9408, 48, 1), 0); del buf119  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf121, (128, 196, 48), (9408, 48, 1), 0), out=buf122)
        buf123 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf122, buf123, 1204224, grid=grid(1204224), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (1568, 768), (768, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf123, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), out=buf124)
        buf125 = reinterpret_tensor(buf124, (8, 196, 768), (150528, 768, 1), 0); del buf124  # reuse
        buf129 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf130 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf401 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47, x_52, x_53, x_54], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf125, buf93, buf101, primals_94, primals_100, primals_21, primals_22, buf129, buf130, buf401, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_100
        del primals_22
        del primals_94
        buf131 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, buf130, reinterpret_tensor(primals_101, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf131)
        del primals_102
        buf132 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55, x_58], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf131, buf132, 4816896, grid=grid(4816896), stream=stream0)
        buf133 = reinterpret_tensor(buf93, (1568, 768), (768, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf132, reinterpret_tensor(primals_103, (3072, 768), (1, 3072), 0), out=buf133)
        buf137 = reinterpret_tensor(buf101, (8, 196, 768), (150528, 768, 1), 0); del buf101  # reuse
        buf138 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf400 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_attn_qk, x_61, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf125, buf133, primals_104, primals_23, primals_24, buf137, buf138, buf400, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_24
        buf139 = buf107; del buf107  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf138, reinterpret_tensor(primals_105, (768, 1536), (1, 768), 0), out=buf139)
        buf140 = reinterpret_tensor(buf111, (307328, 16), (16, 1), 0); del buf111  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_106, (3, 16), (1, 3), 0), out=buf140)
        del primals_106
        buf141 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf139, buf141, 1204224, grid=grid(1204224), stream=stream0)
        buf142 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf139, buf142, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf143 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf142, (128, 48, 196), (9408, 196, 1), 0), out=buf143)
        buf149 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_14], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf140, primals_107, buf149, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_107
        buf146 = reinterpret_tensor(buf140, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf140  # reuse
        buf150 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf152 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_16, attn_17, mul_13, mul_14, patch_score_8, patch_score_9, sigmoid_8, sub_9, sum_5], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf143, primals_25, buf149, buf146, buf150, buf152, 25088, 196, grid=grid(25088), stream=stream0)
        buf151 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf138, reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), out=buf151)
        buf153 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf151, buf153, 1204224, grid=grid(1204224), stream=stream0)
        buf154 = reinterpret_tensor(buf151, (128, 196, 48), (9408, 48, 1), 0); del buf151  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf152, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf153, (128, 196, 48), (9408, 48, 1), 0), out=buf154)
        buf155 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf154, buf155, 1204224, grid=grid(1204224), stream=stream0)
        buf156 = reinterpret_tensor(buf154, (1568, 768), (768, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf155, reinterpret_tensor(primals_109, (768, 768), (1, 768), 0), out=buf156)
        buf157 = reinterpret_tensor(buf156, (8, 196, 768), (150528, 768, 1), 0); del buf156  # reuse
        buf161 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf162 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf399 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61, x_66, x_67, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf157, buf125, buf133, primals_104, primals_110, primals_26, primals_27, buf161, buf162, buf399, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_104
        del primals_110
        del primals_27
        buf163 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_112, buf162, reinterpret_tensor(primals_111, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf163)
        del primals_112
        buf164 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69, x_72], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf163, buf164, 4816896, grid=grid(4816896), stream=stream0)
        buf165 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf164, reinterpret_tensor(primals_113, (3072, 768), (1, 3072), 0), out=buf165)
        buf169 = buf125; del buf125  # reuse
        buf170 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf398 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_attn_qk, x_75, x_76], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf157, buf165, primals_114, primals_28, primals_29, buf169, buf170, buf398, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_29
        buf171 = buf139; del buf139  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf170, reinterpret_tensor(primals_115, (768, 1536), (1, 768), 0), out=buf171)
        buf172 = reinterpret_tensor(buf143, (307328, 16), (16, 1), 0); del buf143  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_116, (3, 16), (1, 3), 0), out=buf172)
        del primals_116
        buf173 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf171, buf173, 1204224, grid=grid(1204224), stream=stream0)
        buf174 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf171, buf174, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf175 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf174, (128, 48, 196), (9408, 196, 1), 0), out=buf175)
        buf181 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_17], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf172, primals_117, buf181, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_117
        buf178 = reinterpret_tensor(buf172, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf172  # reuse
        buf182 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf184 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20, attn_21, mul_16, mul_17, patch_score_10, patch_score_11, sigmoid_10, sub_11, sum_6], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf175, primals_30, buf181, buf178, buf182, buf184, 25088, 196, grid=grid(25088), stream=stream0)
        buf183 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf170, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), out=buf183)
        buf185 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf183, buf185, 1204224, grid=grid(1204224), stream=stream0)
        buf186 = reinterpret_tensor(buf183, (128, 196, 48), (9408, 48, 1), 0); del buf183  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf185, (128, 196, 48), (9408, 48, 1), 0), out=buf186)
        buf187 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf186, buf187, 1204224, grid=grid(1204224), stream=stream0)
        buf188 = reinterpret_tensor(buf186, (1568, 768), (768, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf187, reinterpret_tensor(primals_119, (768, 768), (1, 768), 0), out=buf188)
        buf189 = reinterpret_tensor(buf188, (8, 196, 768), (150528, 768, 1), 0); del buf188  # reuse
        buf193 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf194 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf397 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75, x_80, x_81, x_82], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf189, buf157, buf165, primals_114, primals_120, primals_31, primals_32, buf193, buf194, buf397, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_114
        del primals_120
        del primals_32
        buf195 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_122, buf194, reinterpret_tensor(primals_121, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf195)
        del primals_122
        buf196 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83, x_86], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf195, buf196, 4816896, grid=grid(4816896), stream=stream0)
        buf197 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf196, reinterpret_tensor(primals_123, (3072, 768), (1, 3072), 0), out=buf197)
        buf201 = buf157; del buf157  # reuse
        buf202 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf396 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_attn_qk, x_89, x_90], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf189, buf197, primals_124, primals_33, primals_34, buf201, buf202, buf396, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_34
        buf203 = buf171; del buf171  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, reinterpret_tensor(primals_125, (768, 1536), (1, 768), 0), out=buf203)
        buf204 = reinterpret_tensor(buf175, (307328, 16), (16, 1), 0); del buf175  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_126, (3, 16), (1, 3), 0), out=buf204)
        del primals_126
        buf205 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf203, buf205, 1204224, grid=grid(1204224), stream=stream0)
        buf206 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf203, buf206, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf207 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf206, (128, 48, 196), (9408, 196, 1), 0), out=buf207)
        buf213 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_20], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf204, primals_127, buf213, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_127
        buf210 = reinterpret_tensor(buf204, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf204  # reuse
        buf214 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf216 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_24, attn_25, mul_19, mul_20, patch_score_12, patch_score_13, sigmoid_12, sub_13, sum_7], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf207, primals_35, buf213, buf210, buf214, buf216, 25088, 196, grid=grid(25088), stream=stream0)
        buf215 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf202, reinterpret_tensor(primals_128, (768, 768), (1, 768), 0), out=buf215)
        buf217 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf215, buf217, 1204224, grid=grid(1204224), stream=stream0)
        buf218 = reinterpret_tensor(buf215, (128, 196, 48), (9408, 48, 1), 0); del buf215  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf217, (128, 196, 48), (9408, 48, 1), 0), out=buf218)
        buf219 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf218, buf219, 1204224, grid=grid(1204224), stream=stream0)
        buf220 = reinterpret_tensor(buf218, (1568, 768), (768, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf219, reinterpret_tensor(primals_129, (768, 768), (1, 768), 0), out=buf220)
        buf221 = reinterpret_tensor(buf220, (8, 196, 768), (150528, 768, 1), 0); del buf220  # reuse
        buf225 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf226 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf395 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89, x_94, x_95, x_96], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf221, buf189, buf197, primals_124, primals_130, primals_36, primals_37, buf225, buf226, buf395, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_124
        del primals_130
        del primals_37
        buf227 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf226, reinterpret_tensor(primals_131, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf227)
        del primals_132
        buf228 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100, x_97], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf227, buf228, 4816896, grid=grid(4816896), stream=stream0)
        buf229 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf228, reinterpret_tensor(primals_133, (3072, 768), (1, 3072), 0), out=buf229)
        buf233 = buf189; del buf189  # reuse
        buf234 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf394 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_attn_qk, x_103, x_104], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf221, buf229, primals_134, primals_38, primals_39, buf233, buf234, buf394, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_39
        buf235 = buf203; del buf203  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf234, reinterpret_tensor(primals_135, (768, 1536), (1, 768), 0), out=buf235)
        buf236 = reinterpret_tensor(buf207, (307328, 16), (16, 1), 0); del buf207  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_136, (3, 16), (1, 3), 0), out=buf236)
        del primals_136
        buf237 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf235, buf237, 1204224, grid=grid(1204224), stream=stream0)
        buf238 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf235, buf238, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf239 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf238, (128, 48, 196), (9408, 196, 1), 0), out=buf239)
        buf245 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_23], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf236, primals_137, buf245, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_137
        buf242 = reinterpret_tensor(buf236, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf236  # reuse
        buf246 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf248 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_28, attn_29, mul_22, mul_23, patch_score_14, patch_score_15, sigmoid_14, sub_15, sum_8], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf239, primals_40, buf245, buf242, buf246, buf248, 25088, 196, grid=grid(25088), stream=stream0)
        buf247 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf234, reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), out=buf247)
        buf249 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf247, buf249, 1204224, grid=grid(1204224), stream=stream0)
        buf250 = reinterpret_tensor(buf247, (128, 196, 48), (9408, 48, 1), 0); del buf247  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf249, (128, 196, 48), (9408, 48, 1), 0), out=buf250)
        buf251 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf250, buf251, 1204224, grid=grid(1204224), stream=stream0)
        buf252 = reinterpret_tensor(buf250, (1568, 768), (768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf251, reinterpret_tensor(primals_139, (768, 768), (1, 768), 0), out=buf252)
        buf253 = reinterpret_tensor(buf252, (8, 196, 768), (150528, 768, 1), 0); del buf252  # reuse
        buf257 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf258 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf393 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103, x_108, x_109, x_110], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf253, buf221, buf229, primals_134, primals_140, primals_41, primals_42, buf257, buf258, buf393, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_134
        del primals_140
        del primals_42
        buf259 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_142, buf258, reinterpret_tensor(primals_141, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf259)
        del primals_142
        buf260 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111, x_114], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf259, buf260, 4816896, grid=grid(4816896), stream=stream0)
        buf261 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf260, reinterpret_tensor(primals_143, (3072, 768), (1, 3072), 0), out=buf261)
        buf265 = buf221; del buf221  # reuse
        buf266 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf392 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_attn_qk, x_117, x_118], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf253, buf261, primals_144, primals_43, primals_44, buf265, buf266, buf392, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_44
        buf267 = buf235; del buf235  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf266, reinterpret_tensor(primals_145, (768, 1536), (1, 768), 0), out=buf267)
        buf268 = reinterpret_tensor(buf239, (307328, 16), (16, 1), 0); del buf239  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_146, (3, 16), (1, 3), 0), out=buf268)
        del primals_146
        buf269 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf267, buf269, 1204224, grid=grid(1204224), stream=stream0)
        buf270 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf267, buf270, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf271 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf269, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf270, (128, 48, 196), (9408, 196, 1), 0), out=buf271)
        buf277 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_26], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf268, primals_147, buf277, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_147
        buf274 = reinterpret_tensor(buf268, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf268  # reuse
        buf278 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf280 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_32, attn_33, mul_25, mul_26, patch_score_16, patch_score_17, sigmoid_16, sub_17, sum_9], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf271, primals_45, buf277, buf274, buf278, buf280, 25088, 196, grid=grid(25088), stream=stream0)
        buf279 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf266, reinterpret_tensor(primals_148, (768, 768), (1, 768), 0), out=buf279)
        buf281 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf279, buf281, 1204224, grid=grid(1204224), stream=stream0)
        buf282 = reinterpret_tensor(buf279, (128, 196, 48), (9408, 48, 1), 0); del buf279  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf280, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf281, (128, 196, 48), (9408, 48, 1), 0), out=buf282)
        buf283 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf282, buf283, 1204224, grid=grid(1204224), stream=stream0)
        buf284 = reinterpret_tensor(buf282, (1568, 768), (768, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf283, reinterpret_tensor(primals_149, (768, 768), (1, 768), 0), out=buf284)
        buf285 = reinterpret_tensor(buf284, (8, 196, 768), (150528, 768, 1), 0); del buf284  # reuse
        buf289 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf290 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf391 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117, x_122, x_123, x_124], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf285, buf253, buf261, primals_144, primals_150, primals_46, primals_47, buf289, buf290, buf391, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_144
        del primals_150
        del primals_47
        buf291 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf290, reinterpret_tensor(primals_151, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf291)
        del primals_152
        buf292 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125, x_128], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf291, buf292, 4816896, grid=grid(4816896), stream=stream0)
        buf293 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf292, reinterpret_tensor(primals_153, (3072, 768), (1, 3072), 0), out=buf293)
        buf297 = buf253; del buf253  # reuse
        buf298 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf390 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_attn_qk, x_131, x_132], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf285, buf293, primals_154, primals_48, primals_49, buf297, buf298, buf390, 1568, 768, grid=grid(1568), stream=stream0)
        del primals_49
        buf299 = buf267; del buf267  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf298, reinterpret_tensor(primals_155, (768, 1536), (1, 768), 0), out=buf299)
        buf300 = reinterpret_tensor(buf271, (307328, 16), (16, 1), 0); del buf271  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_pos_proj], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, reinterpret_tensor(primals_156, (3, 16), (1, 3), 0), out=buf300)
        del primals_156
        buf301 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf299, buf301, 1204224, grid=grid(1204224), stream=stream0)
        buf302 = empty((8, 16, 48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf299, buf302, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del buf299
        buf303 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf301, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf302, (128, 48, 196), (9408, 196, 1), 0), out=buf303)
        buf309 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [pos_score_29], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf300, primals_157, buf309, 25088, 196, grid=grid(25088), stream=stream0)
        del primals_157
        buf306 = reinterpret_tensor(buf300, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf300  # reuse
        buf310 = empty((8, 16, 196), device='cuda', dtype=torch.float32)
        buf312 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_36, attn_37, mul_28, mul_29, patch_score_18, patch_score_19, sigmoid_18, sub_19, sum_10], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_add_div_mul_rsub_sigmoid_sum_8.run(buf303, primals_50, buf309, buf306, buf310, buf312, 25088, 196, grid=grid(25088), stream=stream0)
        del buf303
        buf311 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf298, reinterpret_tensor(primals_158, (768, 768), (1, 768), 0), out=buf311)
        buf313 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf311, buf313, 1204224, grid=grid(1204224), stream=stream0)
        buf314 = reinterpret_tensor(buf311, (128, 196, 48), (9408, 48, 1), 0); del buf311  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf312, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf313, (128, 196, 48), (9408, 48, 1), 0), out=buf314)
        buf315 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf314, buf315, 1204224, grid=grid(1204224), stream=stream0)
        buf316 = reinterpret_tensor(buf314, (1568, 768), (768, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf315, reinterpret_tensor(primals_159, (768, 768), (1, 768), 0), out=buf316)
        buf317 = reinterpret_tensor(buf316, (8, 196, 768), (150528, 768, 1), 0); del buf316  # reuse
        buf321 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf322 = empty((1568, 768), device='cuda', dtype=torch.float32)
        buf389 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_131, x_136, x_137, x_138], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf317, buf285, buf293, primals_154, primals_160, primals_51, primals_52, buf321, buf322, buf389, 1568, 768, grid=grid(1568), stream=stream0)
        del buf285
        del primals_154
        del primals_160
        del primals_52
        buf323 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_162, buf322, reinterpret_tensor(primals_161, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf323)
        del primals_162
        buf324 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139, x_142], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf323, buf324, 4816896, grid=grid(4816896), stream=stream0)
        buf325 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf324, reinterpret_tensor(primals_163, (3072, 768), (1, 3072), 0), out=buf325)
        buf326 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf327 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf328 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf330 = reinterpret_tensor(buf328, (8, 197, 1), (197, 1, 1), 0); del buf328  # reuse
        buf331 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, l__mod___blocks_10_attn_qkv, x_147], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_15.run(buf330, primals_2, buf317, buf325, primals_164, primals_53, primals_54, buf326, buf327, buf331, 1576, 768, grid=grid(1576), stream=stream0)
        del buf317
        del buf325
        del primals_164
        del primals_2
        del primals_54
        buf332 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf331, reinterpret_tensor(primals_165, (768, 2304), (1, 768), 0), out=buf332)
        buf333 = empty((8, 16, 197, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf332, buf333, 1210368, grid=grid(1210368), stream=stream0)
        buf334 = empty((8, 16, 48, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf332, buf334, 6144, 197, grid=grid(6144, 197), stream=stream0)
        buf335 = empty((128, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf333, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf334, (128, 48, 197), (9456, 197, 1), 0), out=buf335)
        buf338 = empty((8, 16, 197, 197), device='cuda', dtype=torch.float32)
        buf388 = empty((8, 16, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_40, attn_41, attn_42], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_18.run(buf335, buf338, buf388, 25216, 197, grid=grid(25216), stream=stream0)
        buf339 = empty((8, 16, 197, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf332, buf339, 1210368, grid=grid(1210368), stream=stream0)
        buf340 = empty((128, 197, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf338, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf339, (128, 197, 48), (9456, 48, 1), 0), out=buf340)
        buf341 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.view]
        triton_poi_fused_view_20.run(buf340, buf341, 1210368, grid=grid(1210368), stream=stream0)
        buf342 = reinterpret_tensor(buf340, (1576, 768), (768, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf341, reinterpret_tensor(primals_166, (768, 768), (1, 768), 0), out=buf342)
        buf346 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf347 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf387 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151, x_152, x_153], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_21.run(buf326, buf342, primals_167, primals_55, primals_56, buf346, buf347, buf387, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_56
        buf348 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_169, buf347, reinterpret_tensor(primals_168, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf348)
        del primals_169
        buf349 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154, x_157], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_22.run(buf348, buf349, 4841472, grid=grid(4841472), stream=stream0)
        buf350 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf349, reinterpret_tensor(primals_170, (3072, 768), (1, 3072), 0), out=buf350)
        buf351 = reinterpret_tensor(buf350, (8, 197, 768), (151296, 768, 1), 0); del buf350  # reuse
        buf355 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf356 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf386 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_attn_qkv, x_151, x_160, x_161], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf351, buf326, buf342, primals_167, primals_171, primals_57, primals_58, buf355, buf356, buf386, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_167
        del primals_171
        del primals_58
        buf357 = buf332; del buf332  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(buf356, reinterpret_tensor(primals_172, (768, 2304), (1, 768), 0), out=buf357)
        buf358 = reinterpret_tensor(buf342, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf342  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf357, buf358, 1210368, grid=grid(1210368), stream=stream0)
        buf359 = empty((8, 16, 48, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf357, buf359, 6144, 197, grid=grid(6144, 197), stream=stream0)
        buf360 = buf335; del buf335  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf358, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf359, (128, 48, 197), (9456, 197, 1), 0), out=buf360)
        buf363 = empty((8, 16, 197, 197), device='cuda', dtype=torch.float32)
        buf385 = empty((8, 16, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_43, attn_44, attn_45], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_18.run(buf360, buf363, buf385, 25216, 197, grid=grid(25216), stream=stream0)
        del buf360
        buf364 = empty((8, 16, 197, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf357, buf364, 1210368, grid=grid(1210368), stream=stream0)
        del buf357
        buf365 = empty((128, 197, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf363, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf364, (128, 197, 48), (9456, 48, 1), 0), out=buf365)
        buf366 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.view]
        triton_poi_fused_view_20.run(buf365, buf366, 1210368, grid=grid(1210368), stream=stream0)
        buf367 = reinterpret_tensor(buf365, (1576, 768), (768, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_173, (768, 768), (1, 768), 0), out=buf367)
        buf371 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf372 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf384 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165, x_166, x_167], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_21.run(buf351, buf367, primals_174, primals_59, primals_60, buf371, buf372, buf384, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_60
        buf373 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf372, reinterpret_tensor(primals_175, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf373)
        del primals_176
        buf374 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168, x_171], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_22.run(buf373, buf374, 4841472, grid=grid(4841472), stream=stream0)
        buf375 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf374, reinterpret_tensor(primals_177, (3072, 768), (1, 3072), 0), out=buf375)
        buf376 = reinterpret_tensor(buf375, (8, 197, 768), (151296, 768, 1), 0); del buf375  # reuse
        buf380 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf383 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165, x_174, x_177], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_24.run(buf376, buf351, buf367, primals_174, primals_178, buf380, buf383, 1576, 768, grid=grid(1576), stream=stream0)
        del buf351
        del buf367
        del buf376
        del primals_174
        del primals_178
        buf381 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf380, primals_61, primals_62, buf381, 6144, grid=grid(6144), stream=stream0)
        del primals_62
        buf382 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_180, buf381, reinterpret_tensor(primals_179, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf382)
        del primals_180
        return (buf382, buf8, buf8, buf8, buf8, buf8, buf8, buf8, buf8, buf8, buf8, primals_3, primals_5, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_16, primals_18, primals_20, primals_21, primals_23, primals_25, primals_26, primals_28, primals_30, primals_31, primals_33, primals_35, primals_36, primals_38, primals_40, primals_41, primals_43, primals_45, primals_46, primals_48, primals_50, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_181, buf7, buf9, buf11, buf18, buf21, reinterpret_tensor(buf22, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf27, buf33, buf34, buf35, buf36, buf41, buf42, buf50, buf53, reinterpret_tensor(buf54, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf59, buf65, buf66, buf67, buf68, buf73, buf74, buf82, buf85, reinterpret_tensor(buf86, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf91, buf97, buf98, buf99, buf100, buf105, buf106, buf114, buf117, reinterpret_tensor(buf118, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf123, buf129, buf130, buf131, buf132, buf137, buf138, buf146, buf149, reinterpret_tensor(buf150, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf155, buf161, buf162, buf163, buf164, buf169, buf170, buf178, buf181, reinterpret_tensor(buf182, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf187, buf193, buf194, buf195, buf196, buf201, buf202, buf210, buf213, reinterpret_tensor(buf214, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf219, buf225, buf226, buf227, buf228, buf233, buf234, buf242, buf245, reinterpret_tensor(buf246, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf251, buf257, buf258, buf259, buf260, buf265, buf266, buf274, buf277, reinterpret_tensor(buf278, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf283, buf289, buf290, buf291, buf292, buf297, buf298, buf306, buf309, reinterpret_tensor(buf310, (8, 16, 196, 1), (3136, 196, 1, 1), 0), buf315, buf321, buf322, buf323, buf324, buf326, buf327, buf330, buf331, buf341, buf346, buf347, buf348, buf349, buf355, buf356, buf366, buf371, buf372, buf373, buf374, buf380, buf381, reinterpret_tensor(primals_179, (1000, 768), (768, 1), 0), buf383, reinterpret_tensor(primals_177, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_175, (3072, 768), (768, 1), 0), buf384, reinterpret_tensor(primals_173, (768, 768), (768, 1), 0), reinterpret_tensor(buf363, (128, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf364, (128, 48, 197), (9456, 1, 48), 0), buf385, reinterpret_tensor(buf358, (128, 48, 197), (9456, 1, 48), 0), reinterpret_tensor(buf359, (128, 197, 48), (9456, 1, 197), 0), reinterpret_tensor(primals_172, (2304, 768), (768, 1), 0), buf386, reinterpret_tensor(primals_170, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_168, (3072, 768), (768, 1), 0), buf387, reinterpret_tensor(primals_166, (768, 768), (768, 1), 0), reinterpret_tensor(buf338, (128, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf339, (128, 48, 197), (9456, 1, 48), 0), buf388, reinterpret_tensor(buf333, (128, 48, 197), (9456, 1, 48), 0), reinterpret_tensor(buf334, (128, 197, 48), (9456, 1, 197), 0), reinterpret_tensor(primals_165, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_163, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_161, (3072, 768), (768, 1), 0), buf389, reinterpret_tensor(primals_159, (768, 768), (768, 1), 0), reinterpret_tensor(buf312, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf313, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_158, (768, 768), (768, 1), 0), reinterpret_tensor(buf301, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf302, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_155, (1536, 768), (768, 1), 0), buf390, reinterpret_tensor(primals_153, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_151, (3072, 768), (768, 1), 0), buf391, reinterpret_tensor(primals_149, (768, 768), (768, 1), 0), reinterpret_tensor(buf280, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf281, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_148, (768, 768), (768, 1), 0), reinterpret_tensor(buf269, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf270, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_145, (1536, 768), (768, 1), 0), buf392, reinterpret_tensor(primals_143, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_141, (3072, 768), (768, 1), 0), buf393, reinterpret_tensor(primals_139, (768, 768), (768, 1), 0), reinterpret_tensor(buf248, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf249, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), reinterpret_tensor(buf237, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf238, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_135, (1536, 768), (768, 1), 0), buf394, reinterpret_tensor(primals_133, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 768), (768, 1), 0), buf395, reinterpret_tensor(primals_129, (768, 768), (768, 1), 0), reinterpret_tensor(buf216, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf217, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_128, (768, 768), (768, 1), 0), reinterpret_tensor(buf205, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf206, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_125, (1536, 768), (768, 1), 0), buf396, reinterpret_tensor(primals_123, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_121, (3072, 768), (768, 1), 0), buf397, reinterpret_tensor(primals_119, (768, 768), (768, 1), 0), reinterpret_tensor(buf184, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf185, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), reinterpret_tensor(buf173, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf174, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_115, (1536, 768), (768, 1), 0), buf398, reinterpret_tensor(primals_113, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_111, (3072, 768), (768, 1), 0), buf399, reinterpret_tensor(primals_109, (768, 768), (768, 1), 0), reinterpret_tensor(buf152, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf153, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_108, (768, 768), (768, 1), 0), reinterpret_tensor(buf141, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf142, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_105, (1536, 768), (768, 1), 0), buf400, reinterpret_tensor(primals_103, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_101, (3072, 768), (768, 1), 0), buf401, reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(buf120, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf121, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_98, (768, 768), (768, 1), 0), reinterpret_tensor(buf109, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf110, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_95, (1536, 768), (768, 1), 0), buf402, reinterpret_tensor(primals_93, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 768), (768, 1), 0), buf403, reinterpret_tensor(primals_89, (768, 768), (768, 1), 0), reinterpret_tensor(buf88, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf89, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_88, (768, 768), (768, 1), 0), reinterpret_tensor(buf77, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf78, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_85, (1536, 768), (768, 1), 0), buf404, reinterpret_tensor(primals_83, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_81, (3072, 768), (768, 1), 0), buf405, reinterpret_tensor(primals_79, (768, 768), (768, 1), 0), reinterpret_tensor(buf56, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf57, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_78, (768, 768), (768, 1), 0), reinterpret_tensor(buf45, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf46, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_75, (1536, 768), (768, 1), 0), buf406, reinterpret_tensor(primals_73, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 768), (768, 1), 0), buf407, reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), reinterpret_tensor(buf24, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf25, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_68, (768, 768), (768, 1), 0), reinterpret_tensor(buf13, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf14, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_65, (1536, 768), (768, 1), 0), buf408, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convit_base', benchmark_compiled_module)
