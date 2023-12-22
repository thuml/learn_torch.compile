
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


# kernel path: /tmp/torchinductor_youkaichao/ir/cirwn4qznd3kmowo3pbo53qlnrqhcl6heux4yl55nm5ksasd6gwc.py
# Source Nodes: [cat_5], Original ATen: [aten.cat]
# cat_5 => cat
triton_poi_fused_cat_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 962
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y3 = yindex
    y1 = (yindex // 256)
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 962, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((961*y3) + (((-1) + x2) % 961)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr3 + ((961*y0) + (((-1) + x2) % 961)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tl.store(out_ptr0 + (y0 + (256*x2) + (246272*y1)), tmp18, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pg2qdl6lhh6myopkuwvwdz6irv732b4sh4viddzomjfxhjicpy.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___0___attn_qkv, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_l__mod___transformers_0_blocks___0___attn_qkv => view_1
# getattr_l__mod___transformers_0_blocks___0___norm1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_per_fused_native_layer_norm_view_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 7696
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
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 256.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4umxvkxr66k4hfujhfcwnrodkz3jktxmapjg42yrdkgwi55nuh3.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm2, x_10, x_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___transformers_0_blocks___0___norm2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# x_10 => add_3
# x_11 => view_7
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 7696
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cnethzywdu4gn272dslzxh62exowosmtt3t7pbk7gqwkwr2u5end.py
# Source Nodes: [x_12, x_15], Original ATen: [aten.gelu, aten.view]
# x_12 => add_6, erf, mul_4, mul_5, mul_6
# x_15 => view_9
triton_poi_fused_gelu_view_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7880704
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


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5r77hk7caui3j5sijqjhyv5e2le3es2loh64cri7v4fjag3ajs.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___1___attn_qkv, getattr_l__mod___transformers_0_blocks___1___norm1, x_10, x_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___transformers_0_blocks___1___attn_qkv => view_11
# getattr_l__mod___transformers_0_blocks___1___norm1 => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
# x_10 => add_3
# x_17 => add_7
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 7696
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
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cobuixvd2ke2nhuuewc6kmsvc2ttxsw5y4bbi47eop6eifjgpq2n.py
# Source Nodes: [x_34, x_42], Original ATen: [aten.add]
# x_34 => add_17
# x_42 => add_21
triton_poi_fused_add_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1970176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crtyf4uxsyucuvqodcegfo7pa2mswr4l333fh5eafexjnhm4aquo.py
# Source Nodes: [x_47], Original ATen: [aten.convolution]
# x_47 => convolution_1
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 961
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (256*x2) + (246272*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (961*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplmtvdxlwhatr3baevc6q2iyzy76a5coswx2h6nevi6hx2b2oa3.py
# Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___attn_qkv, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_4 => cat_1
# getattr_l__mod___transformers_1_blocks___0___attn_qkv => view_35
# getattr_l__mod___transformers_1_blocks___0___norm1 => add_23, add_24, mul_21, mul_22, rsqrt_6, sub_6, var_mean_6
triton_per_fused_cat_native_layer_norm_view_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 257
    r2 = rindex
    x1 = (xindex // 257)
    x3 = xindex
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + ((256*r2) + (131072*x1) + (((-1) + x0) % 256)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 512, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = 512.0
    tmp36 = tmp34 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp18 - tmp28
    tmp41 = tmp40 * tmp39
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (512*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp39, xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csrwhfn4xhcvvw2j5moszts6qjk42jumwuheuhdt656ymntd2vrc.py
# Source Nodes: [getattr_l__mod___transformers_1_blocks___0___norm2, x_55, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___transformers_1_blocks___0___norm2 => add_26, add_27, mul_23, mul_24, rsqrt_7, sub_7, var_mean_7
# x_55 => add_25
# x_56 => view_41
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnmvkgfk7wu7ecrvnd4hh7zg5xypnx6zbfmumaxwzrcchekvmy6.py
# Source Nodes: [x_57, x_60], Original ATen: [aten.gelu, aten.view]
# x_57 => add_28, erf_3, mul_25, mul_26, mul_27
# x_60 => view_43
triton_poi_fused_gelu_view_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4210688
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


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnjdet22u7nfz6pg4kjtspnqr7c3f46lep7bskapolgpdq6p2ek.py
# Source Nodes: [getattr_l__mod___transformers_1_blocks___1___attn_qkv, getattr_l__mod___transformers_1_blocks___1___norm1, x_55, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___transformers_1_blocks___1___attn_qkv => view_45
# getattr_l__mod___transformers_1_blocks___1___norm1 => add_30, add_31, mul_28, mul_29, rsqrt_8, sub_8, var_mean_8
# x_55 => add_25
# x_62 => add_29
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxguxvjyc4oo62zibin4hgic3ouh5k6r6l7vaxrs6rbtzpris5s.py
# Source Nodes: [x_115, x_123], Original ATen: [aten.add]
# x_115 => add_60
# x_123 => add_64
triton_poi_fused_add_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1052672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6zx6ihbokrepocugo3oaln4apfhlorfyuk3l7enl5rxmpks3zf.py
# Source Nodes: [x_128], Original ATen: [aten.convolution]
# x_128 => convolution_2
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (512*x2) + (131584*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c67wpg5c4oh32mtv77dqtg7l3uc7sgbukw5dlxojvzabg4g7wkqq.py
# Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___attn_qkv, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_3 => cat_2
# getattr_l__mod___transformers_2_blocks___0___attn_qkv => view_99
# getattr_l__mod___transformers_2_blocks___0___norm1 => add_66, add_67, mul_63, mul_64, rsqrt_18, sub_18, var_mean_18
triton_per_fused_cat_native_layer_norm_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 65
    r2 = rindex
    x1 = (xindex // 65)
    x3 = xindex
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 65, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + ((64*r2) + (65536*x1) + (((-1) + x0) % 64)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 1024, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = 1024.0
    tmp36 = tmp34 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp18 - tmp28
    tmp41 = tmp40 * tmp39
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (1024*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp39, xmask)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjvkew242owacckegm7worolx57d4ypjvmzxbghhjnbsgevus76.py
# Source Nodes: [getattr_l__mod___transformers_2_blocks___0___norm2, x_136, x_137], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___transformers_2_blocks___0___norm2 => add_69, add_70, mul_65, mul_66, rsqrt_19, sub_19, var_mean_19
# x_136 => add_68
# x_137 => view_105
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvyu65jt7otgs3uohpta3q2k4xlbmahn5r5u3g3exgsvwhnr5uw.py
# Source Nodes: [x_138, x_141], Original ATen: [aten.gelu, aten.view]
# x_138 => add_71, erf_9, mul_67, mul_68, mul_69
# x_141 => view_107
triton_poi_fused_gelu_view_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2129920
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


# kernel path: /tmp/torchinductor_youkaichao/3l/c3l4xypqzlsanrf5i4vg3cbypnovifnnhwygdjvdgwu5x7phlah3.py
# Source Nodes: [getattr_l__mod___transformers_2_blocks___1___attn_qkv, getattr_l__mod___transformers_2_blocks___1___norm1, x_136, x_143], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___transformers_2_blocks___1___attn_qkv => view_109
# getattr_l__mod___transformers_2_blocks___1___norm1 => add_73, add_74, mul_70, mul_71, rsqrt_20, sub_20, var_mean_20
# x_136 => add_68
# x_143 => add_72
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 1024.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn45s3h3e6zwl72l5k53h5464obupq4jtg7eddy2tiyrtbzdac4h.py
# Source Nodes: [x_184, x_185], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select]
# x_184 => add_94, add_95, clone_40, mul_91, mul_92, rsqrt_26, sub_26, var_mean_26
# x_185 => select
triton_per_fused_native_layer_norm_native_layer_norm_backward_select_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_select_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (66560*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (66560*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (66560*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 1024.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp36, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173 = args
    args.clear()
    assert_size_stride(primals_1, (1, 256, 31, 31), (246016, 961, 31, 1))
    assert_size_stride(primals_2, (1, 1, 256), (256, 256, 1))
    assert_size_stride(primals_3, (256, 3, 14, 14), (588, 196, 14, 1))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (768, 256), (256, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (256, 256), (256, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (1024, 256), (256, 1))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_15, (256, 1024), (1024, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (768, 256), (256, 1))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (256, 256), (256, 1))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (1024, 256), (256, 1))
    assert_size_stride(primals_26, (1024, ), (1, ))
    assert_size_stride(primals_27, (256, 1024), (1024, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (768, 256), (256, 1))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (256, 256), (256, 1))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (1024, 256), (256, 1))
    assert_size_stride(primals_38, (1024, ), (1, ))
    assert_size_stride(primals_39, (256, 1024), (1024, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256), (256, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (1536, 512), (512, 1))
    assert_size_stride(primals_48, (1536, ), (1, ))
    assert_size_stride(primals_49, (512, 512), (512, 1))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (2048, 512), (512, 1))
    assert_size_stride(primals_54, (2048, ), (1, ))
    assert_size_stride(primals_55, (512, 2048), (2048, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (1536, 512), (512, 1))
    assert_size_stride(primals_60, (1536, ), (1, ))
    assert_size_stride(primals_61, (512, 512), (512, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (2048, 512), (512, 1))
    assert_size_stride(primals_66, (2048, ), (1, ))
    assert_size_stride(primals_67, (512, 2048), (2048, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (1536, 512), (512, 1))
    assert_size_stride(primals_72, (1536, ), (1, ))
    assert_size_stride(primals_73, (512, 512), (512, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (2048, 512), (512, 1))
    assert_size_stride(primals_78, (2048, ), (1, ))
    assert_size_stride(primals_79, (512, 2048), (2048, 1))
    assert_size_stride(primals_80, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (1536, 512), (512, 1))
    assert_size_stride(primals_84, (1536, ), (1, ))
    assert_size_stride(primals_85, (512, 512), (512, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (2048, 512), (512, 1))
    assert_size_stride(primals_90, (2048, ), (1, ))
    assert_size_stride(primals_91, (512, 2048), (2048, 1))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (1536, 512), (512, 1))
    assert_size_stride(primals_96, (1536, ), (1, ))
    assert_size_stride(primals_97, (512, 512), (512, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (2048, 512), (512, 1))
    assert_size_stride(primals_102, (2048, ), (1, ))
    assert_size_stride(primals_103, (512, 2048), (2048, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (1536, 512), (512, 1))
    assert_size_stride(primals_108, (1536, ), (1, ))
    assert_size_stride(primals_109, (512, 512), (512, 1))
    assert_size_stride(primals_110, (512, ), (1, ))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_112, (512, ), (1, ))
    assert_size_stride(primals_113, (2048, 512), (512, 1))
    assert_size_stride(primals_114, (2048, ), (1, ))
    assert_size_stride(primals_115, (512, 2048), (2048, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (1024, ), (1, ))
    assert_size_stride(primals_119, (1024, 512), (512, 1))
    assert_size_stride(primals_120, (1024, ), (1, ))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (1024, ), (1, ))
    assert_size_stride(primals_123, (3072, 1024), (1024, 1))
    assert_size_stride(primals_124, (3072, ), (1, ))
    assert_size_stride(primals_125, (1024, 1024), (1024, 1))
    assert_size_stride(primals_126, (1024, ), (1, ))
    assert_size_stride(primals_127, (1024, ), (1, ))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (4096, 1024), (1024, 1))
    assert_size_stride(primals_130, (4096, ), (1, ))
    assert_size_stride(primals_131, (1024, 4096), (4096, 1))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_135, (3072, 1024), (1024, 1))
    assert_size_stride(primals_136, (3072, ), (1, ))
    assert_size_stride(primals_137, (1024, 1024), (1024, 1))
    assert_size_stride(primals_138, (1024, ), (1, ))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (1024, ), (1, ))
    assert_size_stride(primals_141, (4096, 1024), (1024, 1))
    assert_size_stride(primals_142, (4096, ), (1, ))
    assert_size_stride(primals_143, (1024, 4096), (4096, 1))
    assert_size_stride(primals_144, (1024, ), (1, ))
    assert_size_stride(primals_145, (1024, ), (1, ))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (3072, 1024), (1024, 1))
    assert_size_stride(primals_148, (3072, ), (1, ))
    assert_size_stride(primals_149, (1024, 1024), (1024, 1))
    assert_size_stride(primals_150, (1024, ), (1, ))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_152, (1024, ), (1, ))
    assert_size_stride(primals_153, (4096, 1024), (1024, 1))
    assert_size_stride(primals_154, (4096, ), (1, ))
    assert_size_stride(primals_155, (1024, 4096), (4096, 1))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (1024, ), (1, ))
    assert_size_stride(primals_158, (1024, ), (1, ))
    assert_size_stride(primals_159, (3072, 1024), (1024, 1))
    assert_size_stride(primals_160, (3072, ), (1, ))
    assert_size_stride(primals_161, (1024, 1024), (1024, 1))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_165, (4096, 1024), (1024, 1))
    assert_size_stride(primals_166, (4096, ), (1, ))
    assert_size_stride(primals_167, (1024, 4096), (4096, 1))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (1024, ), (1, ))
    assert_size_stride(primals_171, (1000, 1024), (1024, 1))
    assert_size_stride(primals_172, (1000, ), (1, ))
    assert_size_stride(primals_173, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_173, primals_3, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 256, 31, 31), (246016, 961, 31, 1))
        buf1 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5], Original ATen: [aten.cat]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_cat_0.run(primals_2, buf0, primals_4, primals_1, buf1, 2048, 962, grid=grid(2048, 962), stream=stream0)
        del primals_1
        del primals_2
        del primals_4
        buf2 = empty((8, 962, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 962, 1), (962, 1, 7696), device='cuda', dtype=torch.float32)
        buf5 = reinterpret_tensor(buf3, (8, 962, 1), (962, 1, 1), 0); del buf3  # reuse
        buf6 = empty((7696, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___attn_qkv, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_1.run(buf5, buf1, primals_5, primals_6, buf2, buf6, 7696, 256, grid=grid(7696), stream=stream0)
        del primals_6
        buf7 = empty((7696, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf6, reinterpret_tensor(primals_7, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf7)
        del primals_8
        # Source Nodes: [x_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf8 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, True)
        buf9 = buf8[0]
        buf10 = buf8[1]
        buf11 = buf8[2]
        buf12 = buf8[3]
        del buf8
        buf13 = empty((7696, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (7696, 256), (256, 1), 0), reinterpret_tensor(primals_9, (256, 256), (1, 256), 0), out=buf13)
        buf17 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        buf18 = empty((7696, 256), device='cuda', dtype=torch.float32)
        buf312 = empty((8, 962, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm2, x_10, x_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_2.run(buf1, buf13, primals_10, primals_11, primals_12, buf17, buf18, buf312, 7696, 256, grid=grid(7696), stream=stream0)
        del primals_12
        buf19 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, buf18, reinterpret_tensor(primals_13, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf19)
        del primals_14
        buf20 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_15], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_3.run(buf19, buf20, 7880704, grid=grid(7880704), stream=stream0)
        buf21 = empty((7696, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_15, (1024, 256), (1, 1024), 0), out=buf21)
        buf22 = reinterpret_tensor(buf21, (8, 962, 256), (246272, 256, 1), 0); del buf21  # reuse
        buf26 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        buf27 = empty((7696, 256), device='cuda', dtype=torch.float32)
        buf311 = empty((8, 962, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___attn_qkv, getattr_l__mod___transformers_0_blocks___1___norm1, x_10, x_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4.run(buf22, buf1, buf13, primals_10, primals_16, primals_17, primals_18, buf26, buf27, buf311, 7696, 256, grid=grid(7696), stream=stream0)
        del primals_10
        del primals_16
        del primals_18
        buf28 = empty((7696, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_20, buf27, reinterpret_tensor(primals_19, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf28)
        del primals_20
        # Source Nodes: [x_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf29 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, True)
        buf30 = buf29[0]
        buf31 = buf29[1]
        buf32 = buf29[2]
        buf33 = buf29[3]
        del buf29
        buf34 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (7696, 256), (256, 1), 0), reinterpret_tensor(primals_21, (256, 256), (1, 256), 0), out=buf34)
        buf38 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        buf39 = empty((7696, 256), device='cuda', dtype=torch.float32)
        buf310 = empty((8, 962, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm2, x_22, x_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_2.run(buf22, buf34, primals_22, primals_23, primals_24, buf38, buf39, buf310, 7696, 256, grid=grid(7696), stream=stream0)
        del primals_24
        buf40 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_26, buf39, reinterpret_tensor(primals_25, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf40)
        del primals_26
        buf41 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24, x_27], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_3.run(buf40, buf41, 7880704, grid=grid(7880704), stream=stream0)
        buf42 = empty((7696, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_27, (1024, 256), (1, 1024), 0), out=buf42)
        buf43 = reinterpret_tensor(buf42, (8, 962, 256), (246272, 256, 1), 0); del buf42  # reuse
        buf47 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        buf48 = empty((7696, 256), device='cuda', dtype=torch.float32)
        buf309 = empty((8, 962, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___attn_qkv, getattr_l__mod___transformers_0_blocks___2___norm1, x_22, x_29], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4.run(buf43, buf22, buf34, primals_22, primals_28, primals_29, primals_30, buf47, buf48, buf309, 7696, 256, grid=grid(7696), stream=stream0)
        del primals_22
        del primals_28
        del primals_30
        buf49 = empty((7696, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_32, buf48, reinterpret_tensor(primals_31, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf49)
        del primals_32
        # Source Nodes: [x_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf50 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf49, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf49, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf49, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, True)
        buf51 = buf50[0]
        buf52 = buf50[1]
        buf53 = buf50[2]
        buf54 = buf50[3]
        del buf50
        buf55 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (7696, 256), (256, 1), 0), reinterpret_tensor(primals_33, (256, 256), (1, 256), 0), out=buf55)
        buf59 = buf22; del buf22  # reuse
        buf60 = empty((7696, 256), device='cuda', dtype=torch.float32)
        buf308 = empty((8, 962, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___norm2, x_34, x_35], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_2.run(buf43, buf55, primals_34, primals_35, primals_36, buf59, buf60, buf308, 7696, 256, grid=grid(7696), stream=stream0)
        del primals_36
        buf61 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_38, buf60, reinterpret_tensor(primals_37, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf61)
        del primals_38
        buf62 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36, x_39], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_3.run(buf61, buf62, 7880704, grid=grid(7880704), stream=stream0)
        buf63 = empty((7696, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf62, reinterpret_tensor(primals_39, (1024, 256), (1, 1024), 0), out=buf63)
        buf64 = reinterpret_tensor(buf63, (8, 962, 256), (246272, 256, 1), 0); del buf63  # reuse
        # Source Nodes: [x_34, x_42], Original ATen: [aten.add]
        triton_poi_fused_add_5.run(buf64, buf43, buf55, primals_34, primals_40, 1970176, grid=grid(1970176), stream=stream0)
        del buf43
        del buf55
        del primals_34
        del primals_40
        buf65 = buf0; del buf0  # reuse
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf64, buf65, 2048, 961, grid=grid(2048, 961), stream=stream0)
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_41, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf66, (8, 512, 16, 16), (131072, 256, 16, 1))
        del buf65
        buf67 = empty((8, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_tokens_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (8, 256), (246272, 1), 0), reinterpret_tensor(primals_43, (256, 512), (1, 256), 0), out=buf67)
        buf68 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf69 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((8, 257, 1), (257, 1, 2056), device='cuda', dtype=torch.float32)
        buf72 = reinterpret_tensor(buf70, (8, 257, 1), (257, 1, 1), 0); del buf70  # reuse
        buf73 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___attn_qkv, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_7.run(buf72, buf67, primals_44, buf66, primals_42, primals_45, primals_46, buf68, buf69, buf73, 2056, 512, grid=grid(2056), stream=stream0)
        del buf67
        del primals_42
        del primals_44
        del primals_46
        buf74 = empty((2056, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, buf73, reinterpret_tensor(primals_47, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf74)
        del primals_48
        # Source Nodes: [x_51], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf75 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, True)
        buf76 = buf75[0]
        buf77 = buf75[1]
        buf78 = buf75[2]
        buf79 = buf75[3]
        del buf75
        buf80 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_49, (512, 512), (1, 512), 0), out=buf80)
        buf84 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf85 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf307 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___norm2, x_55, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf68, buf80, primals_50, primals_51, primals_52, buf84, buf85, buf307, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_52
        buf86 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_54, buf85, reinterpret_tensor(primals_53, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf86)
        del primals_54
        buf87 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57, x_60], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf86, buf87, 4210688, grid=grid(4210688), stream=stream0)
        buf88 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf87, reinterpret_tensor(primals_55, (2048, 512), (1, 2048), 0), out=buf88)
        buf89 = reinterpret_tensor(buf88, (8, 257, 512), (131584, 512, 1), 0); del buf88  # reuse
        buf93 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf94 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf306 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___attn_qkv, getattr_l__mod___transformers_1_blocks___1___norm1, x_55, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf89, buf68, buf80, primals_50, primals_56, primals_57, primals_58, buf93, buf94, buf306, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_50
        del primals_56
        del primals_58
        buf95 = empty((2056, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, buf94, reinterpret_tensor(primals_59, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf95)
        del primals_60
        # Source Nodes: [x_63], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf96 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf95, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf95, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf95, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, True)
        buf97 = buf96[0]
        buf98 = buf96[1]
        buf99 = buf96[2]
        buf100 = buf96[3]
        del buf96
        buf101 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_61, (512, 512), (1, 512), 0), out=buf101)
        buf105 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf106 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf305 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___norm2, x_67, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf89, buf101, primals_62, primals_63, primals_64, buf105, buf106, buf305, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_64
        buf107 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_66, buf106, reinterpret_tensor(primals_65, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf107)
        del primals_66
        buf108 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69, x_72], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf107, buf108, 4210688, grid=grid(4210688), stream=stream0)
        buf109 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf108, reinterpret_tensor(primals_67, (2048, 512), (1, 2048), 0), out=buf109)
        buf110 = reinterpret_tensor(buf109, (8, 257, 512), (131584, 512, 1), 0); del buf109  # reuse
        buf114 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf115 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf304 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___attn_qkv, getattr_l__mod___transformers_1_blocks___2___norm1, x_67, x_74], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf110, buf89, buf101, primals_62, primals_68, primals_69, primals_70, buf114, buf115, buf304, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_62
        del primals_68
        del primals_70
        buf116 = empty((2056, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_72, buf115, reinterpret_tensor(primals_71, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf116)
        del primals_72
        # Source Nodes: [x_75], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf117 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, True)
        buf118 = buf117[0]
        buf119 = buf117[1]
        buf120 = buf117[2]
        buf121 = buf117[3]
        del buf117
        buf122 = reinterpret_tensor(buf89, (2056, 512), (512, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_73, (512, 512), (1, 512), 0), out=buf122)
        buf126 = reinterpret_tensor(buf101, (8, 257, 512), (131584, 512, 1), 0); del buf101  # reuse
        buf127 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf303 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___norm2, x_79, x_80], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf110, buf122, primals_74, primals_75, primals_76, buf126, buf127, buf303, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_76
        buf128 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_78, buf127, reinterpret_tensor(primals_77, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf128)
        del primals_78
        buf129 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81, x_84], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf128, buf129, 4210688, grid=grid(4210688), stream=stream0)
        buf130 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf129, reinterpret_tensor(primals_79, (2048, 512), (1, 2048), 0), out=buf130)
        buf131 = reinterpret_tensor(buf130, (8, 257, 512), (131584, 512, 1), 0); del buf130  # reuse
        buf135 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf136 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf302 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___attn_qkv, getattr_l__mod___transformers_1_blocks___3___norm1, x_79, x_86], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf131, buf110, buf122, primals_74, primals_80, primals_81, primals_82, buf135, buf136, buf302, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_74
        del primals_80
        del primals_82
        buf137 = empty((2056, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_84, buf136, reinterpret_tensor(primals_83, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf137)
        del primals_84
        # Source Nodes: [x_87], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf138 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf137, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf137, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf137, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, True)
        buf139 = buf138[0]
        buf140 = buf138[1]
        buf141 = buf138[2]
        buf142 = buf138[3]
        del buf138
        buf143 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_85, (512, 512), (1, 512), 0), out=buf143)
        buf147 = buf110; del buf110  # reuse
        buf148 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf301 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___norm2, x_91, x_92], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf131, buf143, primals_86, primals_87, primals_88, buf147, buf148, buf301, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_88
        buf149 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_90, buf148, reinterpret_tensor(primals_89, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf149)
        del primals_90
        buf150 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93, x_96], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf149, buf150, 4210688, grid=grid(4210688), stream=stream0)
        buf151 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf150, reinterpret_tensor(primals_91, (2048, 512), (1, 2048), 0), out=buf151)
        buf152 = reinterpret_tensor(buf151, (8, 257, 512), (131584, 512, 1), 0); del buf151  # reuse
        buf156 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf157 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf300 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___attn_qkv, getattr_l__mod___transformers_1_blocks___4___norm1, x_91, x_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf152, buf131, buf143, primals_86, primals_92, primals_93, primals_94, buf156, buf157, buf300, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_86
        del primals_92
        del primals_94
        buf158 = empty((2056, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf157, reinterpret_tensor(primals_95, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf158)
        del primals_96
        # Source Nodes: [x_99], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf159 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf158, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf158, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf158, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, True)
        buf160 = buf159[0]
        buf161 = buf159[1]
        buf162 = buf159[2]
        buf163 = buf159[3]
        del buf159
        buf164 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (1, 512), 0), out=buf164)
        buf168 = buf131; del buf131  # reuse
        buf169 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf299 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___norm2, x_103, x_104], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf152, buf164, primals_98, primals_99, primals_100, buf168, buf169, buf299, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_100
        buf170 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, buf169, reinterpret_tensor(primals_101, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf170)
        del primals_102
        buf171 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105, x_108], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf170, buf171, 4210688, grid=grid(4210688), stream=stream0)
        buf172 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf171, reinterpret_tensor(primals_103, (2048, 512), (1, 2048), 0), out=buf172)
        buf173 = reinterpret_tensor(buf172, (8, 257, 512), (131584, 512, 1), 0); del buf172  # reuse
        buf177 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        buf178 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf298 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___attn_qkv, getattr_l__mod___transformers_1_blocks___5___norm1, x_103, x_110], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf173, buf152, buf164, primals_98, primals_104, primals_105, primals_106, buf177, buf178, buf298, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_104
        del primals_106
        del primals_98
        buf179 = empty((2056, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_108, buf178, reinterpret_tensor(primals_107, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf179)
        del primals_108
        # Source Nodes: [x_111], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf180 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf179, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf179, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf179, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, True)
        buf181 = buf180[0]
        buf182 = buf180[1]
        buf183 = buf180[2]
        buf184 = buf180[3]
        del buf180
        buf185 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 512), (1, 512), 0), out=buf185)
        buf189 = buf152; del buf152  # reuse
        buf190 = empty((2056, 512), device='cuda', dtype=torch.float32)
        buf297 = empty((8, 257, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___norm2, x_115, x_116], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf173, buf185, primals_110, primals_111, primals_112, buf189, buf190, buf297, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_112
        buf191 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_114, buf190, reinterpret_tensor(primals_113, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf191)
        del primals_114
        buf192 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117, x_120], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_9.run(buf191, buf192, 4210688, grid=grid(4210688), stream=stream0)
        buf193 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf192, reinterpret_tensor(primals_115, (2048, 512), (1, 2048), 0), out=buf193)
        buf194 = reinterpret_tensor(buf193, (8, 257, 512), (131584, 512, 1), 0); del buf193  # reuse
        # Source Nodes: [x_115, x_123], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf194, buf173, buf185, primals_110, primals_116, 1052672, grid=grid(1052672), stream=stream0)
        del buf173
        del buf185
        del primals_110
        del primals_116
        buf195 = buf66; del buf66  # reuse
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf194, buf195, 4096, 256, grid=grid(4096, 256), stream=stream0)
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_117, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf196, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del buf195
        buf197 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_tokens_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (8, 512), (131584, 1), 0), reinterpret_tensor(primals_119, (512, 1024), (1, 512), 0), out=buf197)
        buf198 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        buf199 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        buf200 = empty_strided((8, 65, 1), (65, 1, 520), device='cuda', dtype=torch.float32)
        buf202 = reinterpret_tensor(buf200, (8, 65, 1), (65, 1, 1), 0); del buf200  # reuse
        buf203 = empty((520, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___attn_qkv, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_13.run(buf202, buf197, primals_120, buf196, primals_118, primals_121, primals_122, buf198, buf199, buf203, 520, 1024, grid=grid(520), stream=stream0)
        del buf196
        del primals_118
        del primals_120
        del primals_122
        buf204 = empty((520, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_124, buf203, reinterpret_tensor(primals_123, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf204)
        del primals_124
        # Source Nodes: [x_132], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf205 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf204, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf204, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf204, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, True)
        buf206 = buf205[0]
        buf207 = buf205[1]
        buf208 = buf205[2]
        buf209 = buf205[3]
        del buf205
        buf210 = empty((520, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_125, (1024, 1024), (1, 1024), 0), out=buf210)
        buf214 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        buf215 = empty((520, 1024), device='cuda', dtype=torch.float32)
        buf296 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___norm2, x_136, x_137], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf198, buf210, primals_126, primals_127, primals_128, buf214, buf215, buf296, 520, 1024, grid=grid(520), stream=stream0)
        del primals_128
        buf216 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_130, buf215, reinterpret_tensor(primals_129, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf216)
        del primals_130
        buf217 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138, x_141], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_15.run(buf216, buf217, 2129920, grid=grid(2129920), stream=stream0)
        buf218 = empty((520, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf217, reinterpret_tensor(primals_131, (4096, 1024), (1, 4096), 0), out=buf218)
        buf219 = reinterpret_tensor(buf218, (8, 65, 1024), (66560, 1024, 1), 0); del buf218  # reuse
        buf223 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        buf224 = empty((520, 1024), device='cuda', dtype=torch.float32)
        buf295 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___attn_qkv, getattr_l__mod___transformers_2_blocks___1___norm1, x_136, x_143], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16.run(buf219, buf198, buf210, primals_126, primals_132, primals_133, primals_134, buf223, buf224, buf295, 520, 1024, grid=grid(520), stream=stream0)
        del primals_126
        del primals_132
        del primals_134
        buf225 = empty((520, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, buf224, reinterpret_tensor(primals_135, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf225)
        del primals_136
        # Source Nodes: [x_144], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf226 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf225, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf225, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf225, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, True)
        buf227 = buf226[0]
        buf228 = buf226[1]
        buf229 = buf226[2]
        buf230 = buf226[3]
        del buf226
        buf231 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_137, (1024, 1024), (1, 1024), 0), out=buf231)
        buf235 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        buf236 = empty((520, 1024), device='cuda', dtype=torch.float32)
        buf294 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___norm2, x_148, x_149], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf219, buf231, primals_138, primals_139, primals_140, buf235, buf236, buf294, 520, 1024, grid=grid(520), stream=stream0)
        del primals_140
        buf237 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_142, buf236, reinterpret_tensor(primals_141, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf237)
        del primals_142
        buf238 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150, x_153], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_15.run(buf237, buf238, 2129920, grid=grid(2129920), stream=stream0)
        buf239 = empty((520, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf238, reinterpret_tensor(primals_143, (4096, 1024), (1, 4096), 0), out=buf239)
        buf240 = reinterpret_tensor(buf239, (8, 65, 1024), (66560, 1024, 1), 0); del buf239  # reuse
        buf244 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        buf245 = empty((520, 1024), device='cuda', dtype=torch.float32)
        buf293 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___attn_qkv, getattr_l__mod___transformers_2_blocks___2___norm1, x_148, x_155], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16.run(buf240, buf219, buf231, primals_138, primals_144, primals_145, primals_146, buf244, buf245, buf293, 520, 1024, grid=grid(520), stream=stream0)
        del primals_138
        del primals_144
        del primals_146
        buf246 = empty((520, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_148, buf245, reinterpret_tensor(primals_147, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf246)
        del primals_148
        # Source Nodes: [x_156], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf247 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf246, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf246, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf246, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, True)
        buf248 = buf247[0]
        buf249 = buf247[1]
        buf250 = buf247[2]
        buf251 = buf247[3]
        del buf247
        buf252 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_149, (1024, 1024), (1, 1024), 0), out=buf252)
        buf256 = buf219; del buf219  # reuse
        buf257 = empty((520, 1024), device='cuda', dtype=torch.float32)
        buf292 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___norm2, x_160, x_161], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf240, buf252, primals_150, primals_151, primals_152, buf256, buf257, buf292, 520, 1024, grid=grid(520), stream=stream0)
        del primals_152
        buf258 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_154, buf257, reinterpret_tensor(primals_153, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf258)
        del primals_154
        buf259 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162, x_165], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_15.run(buf258, buf259, 2129920, grid=grid(2129920), stream=stream0)
        buf260 = empty((520, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf259, reinterpret_tensor(primals_155, (4096, 1024), (1, 4096), 0), out=buf260)
        buf261 = reinterpret_tensor(buf260, (8, 65, 1024), (66560, 1024, 1), 0); del buf260  # reuse
        buf265 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        buf266 = empty((520, 1024), device='cuda', dtype=torch.float32)
        buf291 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___attn_qkv, getattr_l__mod___transformers_2_blocks___3___norm1, x_160, x_167], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16.run(buf261, buf240, buf252, primals_150, primals_156, primals_157, primals_158, buf265, buf266, buf291, 520, 1024, grid=grid(520), stream=stream0)
        del primals_150
        del primals_156
        del primals_158
        buf267 = empty((520, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_160, buf266, reinterpret_tensor(primals_159, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf267)
        del primals_160
        # Source Nodes: [x_168], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf268 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf267, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf267, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf267, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, True)
        buf269 = buf268[0]
        buf270 = buf268[1]
        buf271 = buf268[2]
        buf272 = buf268[3]
        del buf268
        buf273 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_161, (1024, 1024), (1, 1024), 0), out=buf273)
        buf277 = buf240; del buf240  # reuse
        buf278 = empty((520, 1024), device='cuda', dtype=torch.float32)
        buf290 = empty((8, 65, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___norm2, x_172, x_173], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf261, buf273, primals_162, primals_163, primals_164, buf277, buf278, buf290, 520, 1024, grid=grid(520), stream=stream0)
        del primals_164
        buf279 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_166, buf278, reinterpret_tensor(primals_165, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf279)
        del primals_166
        buf280 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174, x_177], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_15.run(buf279, buf280, 2129920, grid=grid(2129920), stream=stream0)
        buf281 = empty((520, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf280, reinterpret_tensor(primals_167, (4096, 1024), (1, 4096), 0), out=buf281)
        buf282 = reinterpret_tensor(buf197, (8, 1, 1024), (1024, 8192, 1), 0); del buf197  # reuse
        buf286 = empty((8, 1, 1024), device='cuda', dtype=torch.float32)
        buf287 = empty((8, 1024), device='cuda', dtype=torch.float32)
        buf289 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184, x_185], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_17.run(buf261, buf273, primals_162, buf281, primals_168, primals_169, primals_170, buf282, buf286, buf287, buf289, 8, 1024, grid=grid(8), stream=stream0)
        del buf261
        del buf273
        del buf281
        del buf282
        del primals_162
        del primals_168
        del primals_170
        buf288 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_172, buf287, reinterpret_tensor(primals_171, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf288)
        del primals_172
        return (buf288, primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_157, primals_163, primals_169, primals_173, buf1, buf2, buf5, buf6, reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 512), buf10, buf11, buf12, reinterpret_tensor(buf9, (7696, 256), (256, 1), 0), buf17, buf18, buf19, buf20, buf26, buf27, reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 512), buf31, buf32, buf33, reinterpret_tensor(buf30, (7696, 256), (256, 1), 0), buf38, buf39, buf40, buf41, buf47, buf48, reinterpret_tensor(buf49, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf49, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf49, (8, 4, 962, 64), (738816, 64, 768, 1), 512), buf52, buf53, buf54, reinterpret_tensor(buf51, (7696, 256), (256, 1), 0), buf59, buf60, buf61, buf62, reinterpret_tensor(buf64, (8, 256, 31, 31), (246272, 1, 7936, 256), 256), reinterpret_tensor(buf64, (8, 256), (246272, 1), 0), buf68, buf69, buf72, buf73, reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf77, buf78, buf79, reinterpret_tensor(buf76, (2056, 512), (512, 1), 0), buf84, buf85, buf86, buf87, buf93, buf94, reinterpret_tensor(buf95, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf95, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf95, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf98, buf99, buf100, reinterpret_tensor(buf97, (2056, 512), (512, 1), 0), buf105, buf106, buf107, buf108, buf114, buf115, reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf119, buf120, buf121, reinterpret_tensor(buf118, (2056, 512), (512, 1), 0), buf126, buf127, buf128, buf129, buf135, buf136, reinterpret_tensor(buf137, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf137, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf137, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf140, buf141, buf142, reinterpret_tensor(buf139, (2056, 512), (512, 1), 0), buf147, buf148, buf149, buf150, buf156, buf157, reinterpret_tensor(buf158, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf158, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf158, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf161, buf162, buf163, reinterpret_tensor(buf160, (2056, 512), (512, 1), 0), buf168, buf169, buf170, buf171, buf177, buf178, reinterpret_tensor(buf179, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf179, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf179, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf182, buf183, buf184, reinterpret_tensor(buf181, (2056, 512), (512, 1), 0), buf189, buf190, buf191, buf192, reinterpret_tensor(buf194, (8, 512, 16, 16), (131584, 1, 8192, 512), 512), reinterpret_tensor(buf194, (8, 512), (131584, 1), 0), buf198, buf199, buf202, buf203, reinterpret_tensor(buf204, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf204, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf204, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf207, buf208, buf209, reinterpret_tensor(buf206, (520, 1024), (1024, 1), 0), buf214, buf215, buf216, buf217, buf223, buf224, reinterpret_tensor(buf225, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf225, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf225, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf228, buf229, buf230, reinterpret_tensor(buf227, (520, 1024), (1024, 1), 0), buf235, buf236, buf237, buf238, buf244, buf245, reinterpret_tensor(buf246, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf246, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf246, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf249, buf250, buf251, reinterpret_tensor(buf248, (520, 1024), (1024, 1), 0), buf256, buf257, buf258, buf259, buf265, buf266, reinterpret_tensor(buf267, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf267, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf267, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf270, buf271, buf272, reinterpret_tensor(buf269, (520, 1024), (1024, 1), 0), buf277, buf278, buf279, buf280, buf286, buf287, reinterpret_tensor(primals_171, (1000, 1024), (1024, 1), 0), buf289, reinterpret_tensor(primals_167, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_165, (4096, 1024), (1024, 1), 0), buf290, reinterpret_tensor(primals_161, (1024, 1024), (1024, 1), 0), buf269, reinterpret_tensor(primals_159, (3072, 1024), (1024, 1), 0), buf291, reinterpret_tensor(primals_155, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_153, (4096, 1024), (1024, 1), 0), buf292, reinterpret_tensor(primals_149, (1024, 1024), (1024, 1), 0), buf248, reinterpret_tensor(primals_147, (3072, 1024), (1024, 1), 0), buf293, reinterpret_tensor(primals_143, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_141, (4096, 1024), (1024, 1), 0), buf294, reinterpret_tensor(primals_137, (1024, 1024), (1024, 1), 0), buf227, reinterpret_tensor(primals_135, (3072, 1024), (1024, 1), 0), buf295, reinterpret_tensor(primals_131, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_129, (4096, 1024), (1024, 1), 0), buf296, reinterpret_tensor(primals_125, (1024, 1024), (1024, 1), 0), buf206, reinterpret_tensor(primals_123, (3072, 1024), (1024, 1), 0), reinterpret_tensor(primals_119, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_115, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_113, (2048, 512), (512, 1), 0), buf297, reinterpret_tensor(primals_109, (512, 512), (512, 1), 0), buf181, reinterpret_tensor(primals_107, (1536, 512), (512, 1), 0), buf298, reinterpret_tensor(primals_103, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_101, (2048, 512), (512, 1), 0), buf299, reinterpret_tensor(primals_97, (512, 512), (512, 1), 0), buf160, reinterpret_tensor(primals_95, (1536, 512), (512, 1), 0), buf300, reinterpret_tensor(primals_91, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_89, (2048, 512), (512, 1), 0), buf301, reinterpret_tensor(primals_85, (512, 512), (512, 1), 0), buf139, reinterpret_tensor(primals_83, (1536, 512), (512, 1), 0), buf302, reinterpret_tensor(primals_79, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_77, (2048, 512), (512, 1), 0), buf303, reinterpret_tensor(primals_73, (512, 512), (512, 1), 0), buf118, reinterpret_tensor(primals_71, (1536, 512), (512, 1), 0), buf304, reinterpret_tensor(primals_67, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_65, (2048, 512), (512, 1), 0), buf305, reinterpret_tensor(primals_61, (512, 512), (512, 1), 0), buf97, reinterpret_tensor(primals_59, (1536, 512), (512, 1), 0), buf306, reinterpret_tensor(primals_55, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_53, (2048, 512), (512, 1), 0), buf307, reinterpret_tensor(primals_49, (512, 512), (512, 1), 0), buf76, reinterpret_tensor(primals_47, (1536, 512), (512, 1), 0), reinterpret_tensor(primals_43, (512, 256), (256, 1), 0), reinterpret_tensor(primals_39, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_37, (1024, 256), (256, 1), 0), buf308, reinterpret_tensor(primals_33, (256, 256), (256, 1), 0), buf51, reinterpret_tensor(primals_31, (768, 256), (256, 1), 0), buf309, reinterpret_tensor(primals_27, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_25, (1024, 256), (256, 1), 0), buf310, reinterpret_tensor(primals_21, (256, 256), (256, 1), 0), buf30, reinterpret_tensor(primals_19, (768, 256), (256, 1), 0), buf311, reinterpret_tensor(primals_15, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_13, (1024, 256), (256, 1), 0), buf312, reinterpret_tensor(primals_9, (256, 256), (256, 1), 0), buf9, reinterpret_tensor(primals_7, (768, 256), (256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 256, 31, 31), (246016, 961, 31, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, 3, 14, 14), (588, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pit_b_224', benchmark_compiled_module)
