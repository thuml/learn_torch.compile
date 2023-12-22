
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


# kernel path: /tmp/torchinductor_youkaichao/sh/cshnf4rz25ni6r4chd4cmhrj7sjixcxg5kzsokaffumfgcr62auo.py
# Source Nodes: [position_ids_1], Original ATen: [aten.view]
# position_ids_1 => view_1
triton_poi_fused_view_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqqa4jzq4gzvbvpyidmjptsu3iezpupqitzkjdyzgkepbszabhi.py
# Source Nodes: [add, hidden_states, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add
# hidden_states => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# inputs_embeds => embedding
# position_embeds => embedding_1
triton_red_fused_add_embedding_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    x0 = xindex % 512
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50257
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50257")
        tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 + 50257
        tmp12 = tmp0 < 0
        tmp13 = tl.where(tmp12, tmp11, tmp0)
        tl.device_assert(((0 <= tmp13) & (tmp13 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp13 < 50257")
        tmp14 = tl.load(in_ptr1 + (r2 + (768*tmp13)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = 768.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-05
        tmp21 = tmp19 + tmp20
        tmp22 = tl.math.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp23, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (768*x3)), tmp27, rmask & xmask)
    tmp28 = 768.0
    tmp29 = tmp9 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp32 / tmp28
    tl.store(out_ptr4 + (x3), tmp33, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqm6kyxe2vnqpnwzbbrzm3mnpif27es5dpxs4uvqoj7io6ca6gp.py
# Source Nodes: [attn_weights], Original ATen: [aten.clone]
# attn_weights => clone_1
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (2304*x1) + (1179648*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/couz6nyby7qvkpmnq3ihbrcosp5k44pmhjyh5fs75mvybm3rwmg3.py
# Source Nodes: [attn_weights], Original ATen: [aten.clone]
# attn_weights => clone_2
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (2304*x2) + (1179648*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cdukjunctqoct2ozlo2kflqwb222kv2y7cmz4nbpzhn7ggvcwqae.py
# Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, attn_weights_6, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
# attn_weights_1 => div
# attn_weights_2 => where
# attn_weights_3 => amax, div_1, exp, sub_1, sum_1
# attn_weights_6 => clone_3
# full => full_default
# mask_value => full_default_1
triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 8.0
        tmp3 = tmp1 / tmp2
        tmp4 = -3.4028234663852886e+38
        tmp5 = tl.where(tmp0, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp10 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = -3.4028234663852886e+38
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tmp14 - tmp7
        tmp16 = tl.exp(tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp21 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp22 = 8.0
        tmp23 = tmp21 / tmp22
        tmp24 = -3.4028234663852886e+38
        tmp25 = tl.where(tmp20, tmp23, tmp24)
        tmp26 = tmp25 - tmp7
        tmp27 = tl.exp(tmp26)
        tmp28 = tmp27 / tmp18
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp28, rmask)
        tl.store(out_ptr3 + (r2 + (512*x3)), tmp28, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cio3atn4h5l726d7ucqhlf3uo23t66chq4av4snadcyytf2643qz.py
# Source Nodes: [attn_output], Original ATen: [aten.clone]
# attn_output => clone_4
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + (64*x2) + (2304*x1) + (1179648*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2vi4j2sjs6aotylrgbdawjv53q6a7r3krwy2je3ycgk3j522f3.py
# Source Nodes: [view_8], Original ATen: [aten.view]
# view_8 => view_14
triton_poi_fused_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nf/cnfqr7qtkcz6kmjdjdhbmxpmapnic7a4nycspttzhv77rx4tb3pg.py
# Source Nodes: [add, hidden_states_2, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add
# hidden_states_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# inputs_embeds => embedding
# position_embeds => embedding_1
# residual_1 => add_3
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + 50257
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp6 < 50257")
    tmp7 = tl.load(in_ptr2 + (r2 + (768*tmp6)), rmask & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 + tmp9
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
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 / tmp28
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp33, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlxidsgbjiphgypyih57rwd3g3ejkkw4gdnnykpybjroacvnrv6.py
# Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_2 => add_6
# add_3 => add_7
# hidden_states_4 => mul_7
# mul => mul_4
# mul_1 => mul_5
# mul_2 => mul_6
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqaxadqazehdb4k7cegjpji4b3nqav2s66lbr654khldek4ftwtm.py
# Source Nodes: [hidden_states_8, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_8 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# residual_2 => add_8
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp24 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnvbcoz7yz5vqflbbdoy3jvtxggrhc5mckjvuqm4olmhbwnjvj4.py
# Source Nodes: [hidden_states_10, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_10 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
# residual_2 => add_8
# residual_3 => add_11
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp2 + tmp7
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
    tmp28 = 1e-05
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162 = args
    args.clear()
    assert_size_stride(primals_1, (2304, ), (1, ))
    assert_size_stride(primals_2, (768, 2304), (2304, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, 768), (768, 1))
    assert_size_stride(primals_5, (3072, ), (1, ))
    assert_size_stride(primals_6, (768, 3072), (3072, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (3072, 768), (768, 1))
    assert_size_stride(primals_9, (2304, ), (1, ))
    assert_size_stride(primals_10, (768, 2304), (2304, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (3072, ), (1, ))
    assert_size_stride(primals_14, (768, 3072), (3072, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (3072, 768), (768, 1))
    assert_size_stride(primals_17, (2304, ), (1, ))
    assert_size_stride(primals_18, (768, 2304), (2304, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, 768), (768, 1))
    assert_size_stride(primals_21, (3072, ), (1, ))
    assert_size_stride(primals_22, (768, 3072), (3072, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (3072, 768), (768, 1))
    assert_size_stride(primals_25, (2304, ), (1, ))
    assert_size_stride(primals_26, (768, 2304), (2304, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, 768), (768, 1))
    assert_size_stride(primals_29, (3072, ), (1, ))
    assert_size_stride(primals_30, (768, 3072), (3072, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (3072, 768), (768, 1))
    assert_size_stride(primals_33, (2304, ), (1, ))
    assert_size_stride(primals_34, (768, 2304), (2304, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, 768), (768, 1))
    assert_size_stride(primals_37, (3072, ), (1, ))
    assert_size_stride(primals_38, (768, 3072), (3072, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (3072, 768), (768, 1))
    assert_size_stride(primals_41, (2304, ), (1, ))
    assert_size_stride(primals_42, (768, 2304), (2304, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, 768), (768, 1))
    assert_size_stride(primals_45, (3072, ), (1, ))
    assert_size_stride(primals_46, (768, 3072), (3072, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (3072, 768), (768, 1))
    assert_size_stride(primals_49, (2304, ), (1, ))
    assert_size_stride(primals_50, (768, 2304), (2304, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, 768), (768, 1))
    assert_size_stride(primals_53, (3072, ), (1, ))
    assert_size_stride(primals_54, (768, 3072), (3072, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (3072, 768), (768, 1))
    assert_size_stride(primals_57, (2304, ), (1, ))
    assert_size_stride(primals_58, (768, 2304), (2304, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, 768), (768, 1))
    assert_size_stride(primals_61, (3072, ), (1, ))
    assert_size_stride(primals_62, (768, 3072), (3072, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (3072, 768), (768, 1))
    assert_size_stride(primals_65, (2304, ), (1, ))
    assert_size_stride(primals_66, (768, 2304), (2304, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, 768), (768, 1))
    assert_size_stride(primals_69, (3072, ), (1, ))
    assert_size_stride(primals_70, (768, 3072), (3072, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (3072, 768), (768, 1))
    assert_size_stride(primals_73, (2304, ), (1, ))
    assert_size_stride(primals_74, (768, 2304), (2304, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, 768), (768, 1))
    assert_size_stride(primals_77, (3072, ), (1, ))
    assert_size_stride(primals_78, (768, 3072), (3072, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (3072, 768), (768, 1))
    assert_size_stride(primals_81, (2304, ), (1, ))
    assert_size_stride(primals_82, (768, 2304), (2304, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, 768), (768, 1))
    assert_size_stride(primals_85, (3072, ), (1, ))
    assert_size_stride(primals_86, (768, 3072), (3072, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (3072, 768), (768, 1))
    assert_size_stride(primals_89, (2304, ), (1, ))
    assert_size_stride(primals_90, (768, 2304), (2304, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (3072, ), (1, ))
    assert_size_stride(primals_94, (768, 3072), (3072, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (3072, 768), (768, 1))
    assert_size_stride(primals_97, (50257, 768), (768, 1))
    assert_size_stride(primals_98, (1024, 768), (768, 1))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (50257, 768), (768, 1))
    assert_size_stride(primals_150, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_151, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_152, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_153, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_154, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_155, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_156, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_157, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_158, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_159, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_160, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_161, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_162, (2, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [position_ids_1], Original ATen: [aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_view_0.run(buf0, 512, grid=grid(512), stream=stream0)
        buf4 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf355 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, hidden_states, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_embedding_native_layer_norm_native_layer_norm_backward_1.run(primals_162, primals_97, primals_98, primals_99, primals_100, buf4, buf5, buf355, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_100
        buf6 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1, reinterpret_tensor(buf5, (1024, 768), (768, 1), 0), primals_2, alpha=1, beta=1, out=buf6)
        del primals_1
        buf7 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf6, buf7, 786432, grid=grid(786432), stream=stream0)
        buf8 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf6, buf8, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf9 = empty((24, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf8, (24, 64, 512), (32768, 512, 1), 0), out=buf9)
        buf12 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf354 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, attn_weights_6, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_150, buf9, buf12, buf354, 12288, 512, grid=grid(12288), stream=stream0)
        buf13 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf6, buf13, 786432, grid=grid(786432), stream=stream0)
        buf14 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf13, (24, 512, 64), (32768, 64, 1), 0), out=buf14)
        buf15 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_8], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf14, buf15, 786432, grid=grid(786432), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (1024, 768), (768, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf15, primals_4, out=buf16)
        buf17 = reinterpret_tensor(buf16, (2, 512, 768), (393216, 768, 1), 0); del buf16  # reuse
        buf21 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf22 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf353 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, hidden_states_2, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_7.run(buf17, primals_3, primals_162, primals_97, primals_98, primals_101, primals_102, buf21, buf22, buf353, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_102
        del primals_3
        del primals_97
        del primals_98
        buf23 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf22, (1024, 768), (768, 1), 0), primals_6, alpha=1, beta=1, out=buf23)
        del primals_5
        buf24 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf25 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf23, buf24, buf25, 3145728, grid=grid(3145728), stream=stream0)
        buf26 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (1024, 3072), (3072, 1), 0), primals_8, out=buf26)
        buf30 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf31 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf352 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf17, buf26, primals_7, primals_103, primals_104, buf30, buf31, buf352, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_104
        buf32 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf31, (1024, 768), (768, 1), 0), primals_10, alpha=1, beta=1, out=buf32)
        del primals_9
        buf33 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf32, buf33, 786432, grid=grid(786432), stream=stream0)
        buf34 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf32, buf34, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf35 = buf9; del buf9  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf34, (24, 64, 512), (32768, 512, 1), 0), out=buf35)
        buf38 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf351 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_10, attn_weights_13, attn_weights_8, attn_weights_9, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_151, buf35, buf38, buf351, 12288, 512, grid=grid(12288), stream=stream0)
        buf39 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf32, buf39, 786432, grid=grid(786432), stream=stream0)
        buf40 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf38, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf39, (24, 512, 64), (32768, 64, 1), 0), out=buf40)
        buf41 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_20], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf40, buf41, 786432, grid=grid(786432), stream=stream0)
        buf42 = reinterpret_tensor(buf40, (1024, 768), (768, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, primals_12, out=buf42)
        buf43 = reinterpret_tensor(buf42, (2, 512, 768), (393216, 768, 1), 0); del buf42  # reuse
        buf47 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf48 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf350 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_10, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf43, primals_11, buf17, buf26, primals_7, primals_105, primals_106, buf47, buf48, buf350, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_106
        del primals_11
        del primals_7
        buf49 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf48, (1024, 768), (768, 1), 0), primals_14, alpha=1, beta=1, out=buf49)
        del primals_13
        buf50 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf51 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, add_7, hidden_states_12, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf49, buf50, buf51, 3145728, grid=grid(3145728), stream=stream0)
        buf52 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 3072), (3072, 1), 0), primals_16, out=buf52)
        buf56 = buf17; del buf17  # reuse
        buf57 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf349 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_16, residual_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf43, buf52, primals_15, primals_107, primals_108, buf56, buf57, buf349, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_108
        buf58 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, reinterpret_tensor(buf57, (1024, 768), (768, 1), 0), primals_18, alpha=1, beta=1, out=buf58)
        del primals_17
        buf59 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf58, buf59, 786432, grid=grid(786432), stream=stream0)
        buf60 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf58, buf60, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf61 = buf35; del buf35  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf60, (24, 64, 512), (32768, 512, 1), 0), out=buf61)
        buf64 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf348 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_15, attn_weights_16, attn_weights_17, attn_weights_20, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_152, buf61, buf64, buf348, 12288, 512, grid=grid(12288), stream=stream0)
        buf65 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf58, buf65, 786432, grid=grid(786432), stream=stream0)
        buf66 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf65, (24, 512, 64), (32768, 64, 1), 0), out=buf66)
        buf67 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_32], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf66, buf67, 786432, grid=grid(786432), stream=stream0)
        buf68 = reinterpret_tensor(buf66, (1024, 768), (768, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf67, primals_20, out=buf68)
        buf69 = reinterpret_tensor(buf68, (2, 512, 768), (393216, 768, 1), 0); del buf68  # reuse
        buf73 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf74 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf347 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf69, primals_19, buf43, buf52, primals_15, primals_109, primals_110, buf73, buf74, buf347, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_110
        del primals_15
        del primals_19
        buf75 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, reinterpret_tensor(buf74, (1024, 768), (768, 1), 0), primals_22, alpha=1, beta=1, out=buf75)
        del primals_21
        buf76 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf77 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, add_11, hidden_states_20, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf75, buf76, buf77, 3145728, grid=grid(3145728), stream=stream0)
        buf78 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (1024, 3072), (3072, 1), 0), primals_24, out=buf78)
        buf82 = buf43; del buf43  # reuse
        buf83 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf346 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24, residual_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf69, buf78, primals_23, primals_111, primals_112, buf82, buf83, buf346, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_112
        buf84 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, reinterpret_tensor(buf83, (1024, 768), (768, 1), 0), primals_26, alpha=1, beta=1, out=buf84)
        del primals_25
        buf85 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf84, buf85, 786432, grid=grid(786432), stream=stream0)
        buf86 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf84, buf86, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf87 = buf61; del buf61  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf86, (24, 64, 512), (32768, 512, 1), 0), out=buf87)
        buf90 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf345 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_22, attn_weights_23, attn_weights_24, attn_weights_27, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_153, buf87, buf90, buf345, 12288, 512, grid=grid(12288), stream=stream0)
        buf91 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf84, buf91, 786432, grid=grid(786432), stream=stream0)
        buf92 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf91, (24, 512, 64), (32768, 64, 1), 0), out=buf92)
        buf93 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_44], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf92, buf93, 786432, grid=grid(786432), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (1024, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf93, primals_28, out=buf94)
        buf95 = reinterpret_tensor(buf94, (2, 512, 768), (393216, 768, 1), 0); del buf94  # reuse
        buf99 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf100 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf344 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_26, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf95, primals_27, buf69, buf78, primals_23, primals_113, primals_114, buf99, buf100, buf344, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_114
        del primals_23
        del primals_27
        buf101 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, reinterpret_tensor(buf100, (1024, 768), (768, 1), 0), primals_30, alpha=1, beta=1, out=buf101)
        del primals_29
        buf102 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf103 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, add_15, hidden_states_28, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf101, buf102, buf103, 3145728, grid=grid(3145728), stream=stream0)
        buf104 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1024, 3072), (3072, 1), 0), primals_32, out=buf104)
        buf108 = buf69; del buf69  # reuse
        buf109 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf343 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, residual_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf95, buf104, primals_31, primals_115, primals_116, buf108, buf109, buf343, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_116
        buf110 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), primals_34, alpha=1, beta=1, out=buf110)
        del primals_33
        buf111 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf110, buf111, 786432, grid=grid(786432), stream=stream0)
        buf112 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf110, buf112, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf113 = buf87; del buf87  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf112, (24, 64, 512), (32768, 512, 1), 0), out=buf113)
        buf116 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf342 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_29, attn_weights_30, attn_weights_31, attn_weights_34, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_154, buf113, buf116, buf342, 12288, 512, grid=grid(12288), stream=stream0)
        buf117 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf110, buf117, 786432, grid=grid(786432), stream=stream0)
        buf118 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf117, (24, 512, 64), (32768, 64, 1), 0), out=buf118)
        buf119 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_56], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf118, buf119, 786432, grid=grid(786432), stream=stream0)
        buf120 = reinterpret_tensor(buf118, (1024, 768), (768, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf119, primals_36, out=buf120)
        buf121 = reinterpret_tensor(buf120, (2, 512, 768), (393216, 768, 1), 0); del buf120  # reuse
        buf125 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf126 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf341 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_34, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf121, primals_35, buf95, buf104, primals_31, primals_117, primals_118, buf125, buf126, buf341, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_118
        del primals_31
        del primals_35
        buf127 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, reinterpret_tensor(buf126, (1024, 768), (768, 1), 0), primals_38, alpha=1, beta=1, out=buf127)
        del primals_37
        buf128 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf129 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19, hidden_states_36, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf127, buf128, buf129, 3145728, grid=grid(3145728), stream=stream0)
        buf130 = reinterpret_tensor(buf95, (1024, 768), (768, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (1024, 3072), (3072, 1), 0), primals_40, out=buf130)
        buf134 = reinterpret_tensor(buf104, (2, 512, 768), (393216, 768, 1), 0); del buf104  # reuse
        buf135 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf340 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_40, residual_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf121, buf130, primals_39, primals_119, primals_120, buf134, buf135, buf340, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_120
        buf136 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_41, reinterpret_tensor(buf135, (1024, 768), (768, 1), 0), primals_42, alpha=1, beta=1, out=buf136)
        del primals_41
        buf137 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf136, buf137, 786432, grid=grid(786432), stream=stream0)
        buf138 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf136, buf138, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf139 = buf113; del buf113  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf138, (24, 64, 512), (32768, 512, 1), 0), out=buf139)
        buf142 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf339 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_36, attn_weights_37, attn_weights_38, attn_weights_41, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_155, buf139, buf142, buf339, 12288, 512, grid=grid(12288), stream=stream0)
        buf143 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf136, buf143, 786432, grid=grid(786432), stream=stream0)
        buf144 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf143, (24, 512, 64), (32768, 64, 1), 0), out=buf144)
        buf145 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_68], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf144, buf145, 786432, grid=grid(786432), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (1024, 768), (768, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf145, primals_44, out=buf146)
        buf147 = reinterpret_tensor(buf146, (2, 512, 768), (393216, 768, 1), 0); del buf146  # reuse
        buf151 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf152 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf338 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_42, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf147, primals_43, buf121, buf130, primals_39, primals_121, primals_122, buf151, buf152, buf338, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_122
        del primals_39
        del primals_43
        buf153 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, reinterpret_tensor(buf152, (1024, 768), (768, 1), 0), primals_46, alpha=1, beta=1, out=buf153)
        del primals_45
        buf154 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf155 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, add_23, hidden_states_44, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf153, buf154, buf155, 3145728, grid=grid(3145728), stream=stream0)
        buf156 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (1024, 3072), (3072, 1), 0), primals_48, out=buf156)
        buf160 = buf121; del buf121  # reuse
        buf161 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf337 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_48, residual_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf147, buf156, primals_47, primals_123, primals_124, buf160, buf161, buf337, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_124
        buf162 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, reinterpret_tensor(buf161, (1024, 768), (768, 1), 0), primals_50, alpha=1, beta=1, out=buf162)
        del primals_49
        buf163 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf162, buf163, 786432, grid=grid(786432), stream=stream0)
        buf164 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf162, buf164, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf165 = buf139; del buf139  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf163, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf164, (24, 64, 512), (32768, 512, 1), 0), out=buf165)
        buf168 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf336 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_43, attn_weights_44, attn_weights_45, attn_weights_48, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_156, buf165, buf168, buf336, 12288, 512, grid=grid(12288), stream=stream0)
        buf169 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf162, buf169, 786432, grid=grid(786432), stream=stream0)
        buf170 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf169, (24, 512, 64), (32768, 64, 1), 0), out=buf170)
        buf171 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_80], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf170, buf171, 786432, grid=grid(786432), stream=stream0)
        buf172 = reinterpret_tensor(buf170, (1024, 768), (768, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, primals_52, out=buf172)
        buf173 = reinterpret_tensor(buf172, (2, 512, 768), (393216, 768, 1), 0); del buf172  # reuse
        buf177 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf178 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf335 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, residual_12, residual_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf173, primals_51, buf147, buf156, primals_47, primals_125, primals_126, buf177, buf178, buf335, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_126
        del primals_47
        del primals_51
        buf179 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_53, reinterpret_tensor(buf178, (1024, 768), (768, 1), 0), primals_54, alpha=1, beta=1, out=buf179)
        del primals_53
        buf180 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf181 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, add_27, hidden_states_52, mul_24, mul_25, mul_26, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf179, buf180, buf181, 3145728, grid=grid(3145728), stream=stream0)
        buf182 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (1024, 3072), (3072, 1), 0), primals_56, out=buf182)
        buf186 = buf147; del buf147  # reuse
        buf187 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf334 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_56, residual_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf173, buf182, primals_55, primals_127, primals_128, buf186, buf187, buf334, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_128
        buf188 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_57, reinterpret_tensor(buf187, (1024, 768), (768, 1), 0), primals_58, alpha=1, beta=1, out=buf188)
        del primals_57
        buf189 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf188, buf189, 786432, grid=grid(786432), stream=stream0)
        buf190 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf188, buf190, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf191 = buf165; del buf165  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf190, (24, 64, 512), (32768, 512, 1), 0), out=buf191)
        buf194 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf333 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_50, attn_weights_51, attn_weights_52, attn_weights_55, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_157, buf191, buf194, buf333, 12288, 512, grid=grid(12288), stream=stream0)
        buf195 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf188, buf195, 786432, grid=grid(786432), stream=stream0)
        buf196 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf195, (24, 512, 64), (32768, 64, 1), 0), out=buf196)
        buf197 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_92], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf196, buf197, 786432, grid=grid(786432), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (1024, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf197, primals_60, out=buf198)
        buf199 = reinterpret_tensor(buf198, (2, 512, 768), (393216, 768, 1), 0); del buf198  # reuse
        buf203 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf204 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf332 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf199, primals_59, buf173, buf182, primals_55, primals_129, primals_130, buf203, buf204, buf332, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_130
        del primals_55
        del primals_59
        buf205 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, reinterpret_tensor(buf204, (1024, 768), (768, 1), 0), primals_62, alpha=1, beta=1, out=buf205)
        del primals_61
        buf206 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf207 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, add_31, hidden_states_60, mul_28, mul_29, mul_30, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf205, buf206, buf207, 3145728, grid=grid(3145728), stream=stream0)
        buf208 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1024, 3072), (3072, 1), 0), primals_64, out=buf208)
        buf212 = buf173; del buf173  # reuse
        buf213 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf331 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_64, residual_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf199, buf208, primals_63, primals_131, primals_132, buf212, buf213, buf331, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_132
        buf214 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, reinterpret_tensor(buf213, (1024, 768), (768, 1), 0), primals_66, alpha=1, beta=1, out=buf214)
        del primals_65
        buf215 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf214, buf215, 786432, grid=grid(786432), stream=stream0)
        buf216 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf214, buf216, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf217 = buf191; del buf191  # reuse
        # Source Nodes: [attn_weights_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf215, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf216, (24, 64, 512), (32768, 512, 1), 0), out=buf217)
        buf220 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf330 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_57, attn_weights_58, attn_weights_59, attn_weights_62, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_158, buf217, buf220, buf330, 12288, 512, grid=grid(12288), stream=stream0)
        buf221 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf214, buf221, 786432, grid=grid(786432), stream=stream0)
        buf222 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf221, (24, 512, 64), (32768, 64, 1), 0), out=buf222)
        buf223 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_104], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf222, buf223, 786432, grid=grid(786432), stream=stream0)
        buf224 = reinterpret_tensor(buf222, (1024, 768), (768, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf223, primals_68, out=buf224)
        buf225 = reinterpret_tensor(buf224, (2, 512, 768), (393216, 768, 1), 0); del buf224  # reuse
        buf229 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf230 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf329 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66, residual_16, residual_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf225, primals_67, buf199, buf208, primals_63, primals_133, primals_134, buf229, buf230, buf329, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_134
        del primals_63
        del primals_67
        buf231 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_69, reinterpret_tensor(buf230, (1024, 768), (768, 1), 0), primals_70, alpha=1, beta=1, out=buf231)
        del primals_69
        buf232 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf233 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_34, add_35, hidden_states_68, mul_32, mul_33, mul_34, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf231, buf232, buf233, 3145728, grid=grid(3145728), stream=stream0)
        buf234 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (1024, 3072), (3072, 1), 0), primals_72, out=buf234)
        buf238 = buf199; del buf199  # reuse
        buf239 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf328 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72, residual_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf225, buf234, primals_71, primals_135, primals_136, buf238, buf239, buf328, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_136
        buf240 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_73, reinterpret_tensor(buf239, (1024, 768), (768, 1), 0), primals_74, alpha=1, beta=1, out=buf240)
        del primals_73
        buf241 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf240, buf241, 786432, grid=grid(786432), stream=stream0)
        buf242 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf240, buf242, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf243 = buf217; del buf217  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf242, (24, 64, 512), (32768, 512, 1), 0), out=buf243)
        buf246 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf327 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_64, attn_weights_65, attn_weights_66, attn_weights_69, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_159, buf243, buf246, buf327, 12288, 512, grid=grid(12288), stream=stream0)
        buf247 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf240, buf247, 786432, grid=grid(786432), stream=stream0)
        buf248 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf247, (24, 512, 64), (32768, 64, 1), 0), out=buf248)
        buf249 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_116], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf248, buf249, 786432, grid=grid(786432), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (1024, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf249, primals_76, out=buf250)
        buf251 = reinterpret_tensor(buf250, (2, 512, 768), (393216, 768, 1), 0); del buf250  # reuse
        buf255 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf256 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf326 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_74, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf251, primals_75, buf225, buf234, primals_71, primals_137, primals_138, buf255, buf256, buf326, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_138
        del primals_71
        del primals_75
        buf257 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, reinterpret_tensor(buf256, (1024, 768), (768, 1), 0), primals_78, alpha=1, beta=1, out=buf257)
        del primals_77
        buf258 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf259 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, add_39, hidden_states_76, mul_36, mul_37, mul_38, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf257, buf258, buf259, 3145728, grid=grid(3145728), stream=stream0)
        buf260 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf259, (1024, 3072), (3072, 1), 0), primals_80, out=buf260)
        buf264 = buf225; del buf225  # reuse
        buf265 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf325 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_80, residual_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf251, buf260, primals_79, primals_139, primals_140, buf264, buf265, buf325, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_140
        buf266 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, reinterpret_tensor(buf265, (1024, 768), (768, 1), 0), primals_82, alpha=1, beta=1, out=buf266)
        del primals_81
        buf267 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf266, buf267, 786432, grid=grid(786432), stream=stream0)
        buf268 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf266, buf268, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf269 = buf243; del buf243  # reuse
        # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf268, (24, 64, 512), (32768, 512, 1), 0), out=buf269)
        buf272 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf324 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_71, attn_weights_72, attn_weights_73, attn_weights_76, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_160, buf269, buf272, buf324, 12288, 512, grid=grid(12288), stream=stream0)
        buf273 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf266, buf273, 786432, grid=grid(786432), stream=stream0)
        buf274 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf272, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf273, (24, 512, 64), (32768, 64, 1), 0), out=buf274)
        buf275 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_128], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf274, buf275, 786432, grid=grid(786432), stream=stream0)
        buf276 = reinterpret_tensor(buf274, (1024, 768), (768, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf275, primals_84, out=buf276)
        buf277 = reinterpret_tensor(buf276, (2, 512, 768), (393216, 768, 1), 0); del buf276  # reuse
        buf281 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf282 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf323 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_82, residual_20, residual_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf277, primals_83, buf251, buf260, primals_79, primals_141, primals_142, buf281, buf282, buf323, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_142
        del primals_79
        del primals_83
        buf283 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_85, reinterpret_tensor(buf282, (1024, 768), (768, 1), 0), primals_86, alpha=1, beta=1, out=buf283)
        del primals_85
        buf284 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf285 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, add_43, hidden_states_84, mul_40, mul_41, mul_42, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf283, buf284, buf285, 3145728, grid=grid(3145728), stream=stream0)
        buf286 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf285, (1024, 3072), (3072, 1), 0), primals_88, out=buf286)
        buf290 = buf251; del buf251  # reuse
        buf291 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf322 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_88, residual_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf277, buf286, primals_87, primals_143, primals_144, buf290, buf291, buf322, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_144
        buf292 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_89, reinterpret_tensor(buf291, (1024, 768), (768, 1), 0), primals_90, alpha=1, beta=1, out=buf292)
        del primals_89
        buf293 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf292, buf293, 786432, grid=grid(786432), stream=stream0)
        buf294 = empty((2, 12, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf292, buf294, 1536, 512, grid=grid(1536, 512), stream=stream0)
        buf295 = buf269; del buf269  # reuse
        # Source Nodes: [attn_weights_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf293, (24, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf294, (24, 64, 512), (32768, 512, 1), 0), out=buf295)
        buf298 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf321 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_78, attn_weights_79, attn_weights_80, attn_weights_83, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.clone, aten.detach, aten.div, aten.full, aten.where]
        triton_red_fused__softmax__to_copy_clone_detach_div_full_where_4.run(primals_161, buf295, buf298, buf321, 12288, 512, grid=grid(12288), stream=stream0)
        del buf295
        buf299 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf292, buf299, 786432, grid=grid(786432), stream=stream0)
        buf300 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf299, (24, 512, 64), (32768, 64, 1), 0), out=buf300)
        buf301 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [view_140], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf300, buf301, 786432, grid=grid(786432), stream=stream0)
        buf302 = reinterpret_tensor(buf300, (1024, 768), (768, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf301, primals_92, out=buf302)
        buf303 = reinterpret_tensor(buf302, (2, 512, 768), (393216, 768, 1), 0); del buf302  # reuse
        buf307 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf308 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        buf320 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf303, primals_91, buf277, buf286, primals_87, primals_145, primals_146, buf307, buf308, buf320, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_146
        del primals_87
        del primals_91
        buf309 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, reinterpret_tensor(buf308, (1024, 768), (768, 1), 0), primals_94, alpha=1, beta=1, out=buf309)
        del primals_93
        buf310 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        buf311 = empty((2, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_46, add_47, hidden_states_92, mul_44, mul_45, mul_46, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_8.run(buf309, buf310, buf311, 3145728, grid=grid(3145728), stream=stream0)
        buf312 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (1024, 3072), (3072, 1), 0), primals_96, out=buf312)
        buf316 = buf277; del buf277  # reuse
        buf317 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf319 = empty((2, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, hidden_states_96, l__mod___transformer_ln_f, lm_logits], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf303, buf312, primals_95, primals_147, primals_148, buf316, buf317, buf319, 1024, 768, grid=grid(1024), stream=stream0)
        del buf303
        del buf312
        del primals_148
        del primals_95
        buf318 = empty((1024, 50257), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(buf317, reinterpret_tensor(primals_149, (768, 50257), (1, 768), 0), out=buf318)
        return (reinterpret_tensor(buf318, (2, 512, 50257), (25731584, 50257, 1), 0), reinterpret_tensor(buf6, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf6, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf32, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf32, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf58, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf58, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf84, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf84, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf110, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf110, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf136, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf136, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf162, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf162, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf188, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf188, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf214, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf214, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf240, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf240, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf266, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf266, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), reinterpret_tensor(buf292, (2, 12, 512, 64), (1179648, 64, 2304, 1), 768), reinterpret_tensor(buf292, (2, 12, 512, 64), (1179648, 64, 2304, 1), 1536), primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_162, buf0, buf4, reinterpret_tensor(primals_150, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf21, buf23, buf24, buf30, reinterpret_tensor(primals_151, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf47, buf49, buf50, buf56, reinterpret_tensor(primals_152, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf73, buf75, buf76, buf82, reinterpret_tensor(primals_153, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf99, buf101, buf102, buf108, reinterpret_tensor(primals_154, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf125, buf127, buf128, buf134, reinterpret_tensor(primals_155, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf151, buf153, buf154, buf160, reinterpret_tensor(primals_156, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf177, buf179, buf180, buf186, reinterpret_tensor(primals_157, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf203, buf205, buf206, buf212, reinterpret_tensor(primals_158, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf229, buf231, buf232, buf238, reinterpret_tensor(primals_159, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf255, buf257, buf258, buf264, reinterpret_tensor(primals_160, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf281, buf283, buf284, buf290, reinterpret_tensor(primals_161, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf307, buf309, buf310, buf316, buf317, reinterpret_tensor(primals_149, (50257, 768), (768, 1), 0), buf319, reinterpret_tensor(primals_96, (768, 3072), (1, 768), 0), reinterpret_tensor(buf311, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_94, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf308, (768, 1024), (1, 768), 0), buf320, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), reinterpret_tensor(buf301, (768, 1024), (1, 768), 0), reinterpret_tensor(buf298, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf299, (24, 64, 512), (32768, 1, 64), 0), buf321, reinterpret_tensor(buf293, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf294, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_90, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf291, (768, 1024), (1, 768), 0), buf322, reinterpret_tensor(primals_88, (768, 3072), (1, 768), 0), reinterpret_tensor(buf285, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_86, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf282, (768, 1024), (1, 768), 0), buf323, reinterpret_tensor(primals_84, (768, 768), (1, 768), 0), reinterpret_tensor(buf275, (768, 1024), (1, 768), 0), reinterpret_tensor(buf272, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf273, (24, 64, 512), (32768, 1, 64), 0), buf324, reinterpret_tensor(buf267, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf268, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_82, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf265, (768, 1024), (1, 768), 0), buf325, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), reinterpret_tensor(buf259, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_78, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf256, (768, 1024), (1, 768), 0), buf326, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), reinterpret_tensor(buf249, (768, 1024), (1, 768), 0), reinterpret_tensor(buf246, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf247, (24, 64, 512), (32768, 1, 64), 0), buf327, reinterpret_tensor(buf241, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf242, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_74, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf239, (768, 1024), (1, 768), 0), buf328, reinterpret_tensor(primals_72, (768, 3072), (1, 768), 0), reinterpret_tensor(buf233, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_70, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf230, (768, 1024), (1, 768), 0), buf329, reinterpret_tensor(primals_68, (768, 768), (1, 768), 0), reinterpret_tensor(buf223, (768, 1024), (1, 768), 0), reinterpret_tensor(buf220, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf221, (24, 64, 512), (32768, 1, 64), 0), buf330, reinterpret_tensor(buf215, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf216, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_66, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf213, (768, 1024), (1, 768), 0), buf331, reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), reinterpret_tensor(buf207, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_62, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf204, (768, 1024), (1, 768), 0), buf332, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), reinterpret_tensor(buf197, (768, 1024), (1, 768), 0), reinterpret_tensor(buf194, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf195, (24, 64, 512), (32768, 1, 64), 0), buf333, reinterpret_tensor(buf189, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf190, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_58, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf187, (768, 1024), (1, 768), 0), buf334, reinterpret_tensor(primals_56, (768, 3072), (1, 768), 0), reinterpret_tensor(buf181, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_54, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf178, (768, 1024), (1, 768), 0), buf335, reinterpret_tensor(primals_52, (768, 768), (1, 768), 0), reinterpret_tensor(buf171, (768, 1024), (1, 768), 0), reinterpret_tensor(buf168, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf169, (24, 64, 512), (32768, 1, 64), 0), buf336, reinterpret_tensor(buf163, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf164, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_50, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf161, (768, 1024), (1, 768), 0), buf337, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), reinterpret_tensor(buf155, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_46, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf152, (768, 1024), (1, 768), 0), buf338, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), reinterpret_tensor(buf145, (768, 1024), (1, 768), 0), reinterpret_tensor(buf142, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf143, (24, 64, 512), (32768, 1, 64), 0), buf339, reinterpret_tensor(buf137, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf138, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_42, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf135, (768, 1024), (1, 768), 0), buf340, reinterpret_tensor(primals_40, (768, 3072), (1, 768), 0), reinterpret_tensor(buf129, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_38, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf126, (768, 1024), (1, 768), 0), buf341, reinterpret_tensor(primals_36, (768, 768), (1, 768), 0), reinterpret_tensor(buf119, (768, 1024), (1, 768), 0), reinterpret_tensor(buf116, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf117, (24, 64, 512), (32768, 1, 64), 0), buf342, reinterpret_tensor(buf111, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf112, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_34, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf109, (768, 1024), (1, 768), 0), buf343, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), reinterpret_tensor(buf103, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_30, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf100, (768, 1024), (1, 768), 0), buf344, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), reinterpret_tensor(buf93, (768, 1024), (1, 768), 0), reinterpret_tensor(buf90, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf91, (24, 64, 512), (32768, 1, 64), 0), buf345, reinterpret_tensor(buf85, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf86, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_26, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf83, (768, 1024), (1, 768), 0), buf346, reinterpret_tensor(primals_24, (768, 3072), (1, 768), 0), reinterpret_tensor(buf77, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_22, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf74, (768, 1024), (1, 768), 0), buf347, reinterpret_tensor(primals_20, (768, 768), (1, 768), 0), reinterpret_tensor(buf67, (768, 1024), (1, 768), 0), reinterpret_tensor(buf64, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf65, (24, 64, 512), (32768, 1, 64), 0), buf348, reinterpret_tensor(buf59, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf60, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_18, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf57, (768, 1024), (1, 768), 0), buf349, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), reinterpret_tensor(buf51, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_14, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf48, (768, 1024), (1, 768), 0), buf350, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), reinterpret_tensor(buf41, (768, 1024), (1, 768), 0), reinterpret_tensor(buf38, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf39, (24, 64, 512), (32768, 1, 64), 0), buf351, reinterpret_tensor(buf33, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf34, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_10, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf31, (768, 1024), (1, 768), 0), buf352, reinterpret_tensor(primals_8, (768, 3072), (1, 768), 0), reinterpret_tensor(buf25, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_6, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf22, (768, 1024), (1, 768), 0), buf353, reinterpret_tensor(primals_4, (768, 768), (1, 768), 0), reinterpret_tensor(buf15, (768, 1024), (1, 768), 0), reinterpret_tensor(buf12, (24, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf13, (24, 64, 512), (32768, 1, 64), 0), buf354, reinterpret_tensor(buf7, (24, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf8, (24, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_2, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf5, (768, 1024), (1, 768), 0), buf355, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_151 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_152 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_153 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_154 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_155 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_156 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_157 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_158 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_159 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_160 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_161 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_162 = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_GPT2', benchmark_compiled_module)
