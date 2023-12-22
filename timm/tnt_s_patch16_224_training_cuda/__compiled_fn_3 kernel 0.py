
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


# kernel path: /tmp/torchinductor_youkaichao/44/c44c3zp6z4q7o66otyzrinaisp5m7ibvnr27whbkm4julr2eqkmy.py
# Source Nodes: [x_1], Original ATen: [aten.im2col]
# x_1 => add
triton_poi_fused_im2col_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_im2col_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    tmp0 = x1 + (4*x0)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/lv/clvl7a6zui34dql7sav26dnsoprngn2rolrprsoq4pafbdax54to.py
# Source Nodes: [x_3], Original ATen: [aten.add]
# x_3 => add_2
triton_poi_fused_add_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 4) % 4
    x3 = (xindex // 384)
    x0 = xindex % 4
    x2 = (xindex // 16) % 24
    x6 = xindex % 384
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*x1) + ((x3 % 196) // 14)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + ((14*x0) + ((x3 % 196) % 14)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 56
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 56), "index out of bounds: 0 <= tmp3 < 56")
    tmp5 = tmp4 + 56
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 56), "index out of bounds: 0 <= tmp7 < 56")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (56*tmp3) + (3136*x2) + (75264*(x3 // 196))), None, eviction_policy='evict_last')
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x7), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfyznnmn27n242rsvtqva2beoj6bycgirtwm6ihndqqflcp74vdt.py
# Source Nodes: [l__mod___blocks_0_norm_in, reshape_2], Original ATen: [aten.clone, aten.native_layer_norm]
# l__mod___blocks_0_norm_in => add_8, rsqrt_2, var_mean_2
# reshape_2 => clone_2
triton_per_fused_clone_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (384*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 24.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.store(out_ptr0 + (r2 + (24*x3)), tmp0, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbdbtks5dvmzx4bvlgkvqn7nthzhijs4iypnvrx4di7tmeyf463.py
# Source Nodes: [l__mod___blocks_0_attn_in_qk, l__mod___blocks_0_norm_in, l__mod___norm1_proj, l__mod___proj], Original ATen: [aten.native_layer_norm, aten.view]
# l__mod___blocks_0_attn_in_qk => view_6
# l__mod___blocks_0_norm_in => add_9, mul_4, mul_5, sub_2
# l__mod___norm1_proj => add_3, add_4, mul, mul_1, rsqrt, sub, var_mean
# l__mod___proj => view_4
triton_per_fused_native_layer_norm_view_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    r3 = (rindex // 24)
    r2 = rindex % 24
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r3 + (16*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r3 + (16*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp17 = 384.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp0 - tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (384*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwndkh2tohdgexqdvwsnqcfs4sqrfpyozfat377tbjlwjwqjjee.py
# Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm]
# patch_embed => add_5, rsqrt_1, var_mean_1
triton_per_fused_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel):
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
    tmp17 = 384.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c42wpixwfalifaij2wwes3zkz6nyz2v3s6zlf4kjfjh2ddhf456b.py
# Source Nodes: [cat_25, patch_embed_2], Original ATen: [aten.add, aten.cat]
# cat_25 => cat
# patch_embed_2 => add_7
triton_poi_fused_add_cat_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_5', 'mutated_arg_names': []},
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
    tmp23 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
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
    tmp15 = tmp13 * tmp14
    tmp16 = tl.load(in_ptr4 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr5 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x4), tmp24, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/sx/csxy5yngjpkautbl5epm6c4rnf3ivs7kryodjzfnb23d3wzc223k.py
# Source Nodes: [x_6], Original ATen: [aten.view]
# x_6 => view_19
triton_poi_fused_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((6*(x1 % 16)) + (96*(x0 // 6)) + (384*(x1 // 16)) + (x0 % 6)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kf/ckfxwlwfsmbenmpilm7bznjf73vki4ffnzcehm7kwxjyktmsqppj.py
# Source Nodes: [l__mod___blocks_0_norm_mlp_in, pixel_embed_1, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_norm_mlp_in => add_11, add_12, clone_9, mul_7, mul_8, rsqrt_3, sub_4, var_mean_3
# pixel_embed_1 => add_10
# x_8 => view_21
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (384*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (24*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r2 + (24*x3)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (24*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4au3wn44nmmum5ufqwps4i7avizthyuomhg6674eeebav7p4non.py
# Source Nodes: [x_12, x_9], Original ATen: [aten.gelu, aten.view]
# x_12 => view_23
# x_9 => add_13, erf, mul_10, mul_11, mul_9
triton_poi_fused_gelu_view_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
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


# kernel path: /tmp/torchinductor_youkaichao/fy/cfy3imzjjbqv5gjyxtil7de7cwxzvuha4ugxpaefypoqpz4fztp3.py
# Source Nodes: [l__mod___blocks_0_norm1_proj, l__mod___blocks_0_proj, l__mod___blocks_1_attn_in_qk, l__mod___blocks_1_norm_in, pixel_embed_1, pixel_embed_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_norm1_proj => add_15, clone_12, mul_12, rsqrt_4, sub_5, var_mean_4
# l__mod___blocks_0_proj => view_26
# l__mod___blocks_1_attn_in_qk => view_47
# l__mod___blocks_1_norm_in => add_26, mul_23
# pixel_embed_1 => add_10
# pixel_embed_3 => add_14
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (384*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (24*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (24*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r2 + (24*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (24*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (24*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (24*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr5 + (x3), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswnwy2hgx73pca3ctsmpfq5pt2g2oydeyoj375eyiirjhqykazd.py
# Source Nodes: [cat_24, l__mod___blocks_0_attn_out_qk, l__mod___blocks_0_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_24 => cat_1
# l__mod___blocks_0_attn_out_qk => view_28
# l__mod___blocks_0_norm_out => add_18, add_19, mul_14, mul_15, rsqrt_5, sub_6, var_mean_5
triton_per_fused_cat_native_layer_norm_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp35 = 384.0
    tmp36 = tmp34 / tmp35
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp18 - tmp28
    tmp41 = tmp40 * tmp39
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (384*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp39, xmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6wttskx4fzn4n723omagnobjiwep7xoaokdfft3hzsp3uacoi5.py
# Source Nodes: [matmul_2], Original ATen: [aten.clone]
# matmul_2 => clone_13
triton_poi_fused_clone_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rg/crg6wbkumtqs6nptmi427nmc4hw6sos2mbwd4fxjbnifj3qchww5.py
# Source Nodes: [matmul_2], Original ATen: [aten.clone]
# matmul_2 => clone_14
triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbns3q74uqxci5z6dy42bd2dsagtzkcczm2osck3hlm3woggkd5.py
# Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
# attn_3 => mul_16
# attn_4 => amax_1, div_1, exp_1, sub_7, sum_2
triton_per_fused__softmax_mul_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/y2/cy26abm34k4gl6h66xz7hbxh2zxhzdntmmxllpfenblqsoegnugq.py
# Source Nodes: [matmul_3], Original ATen: [aten.clone]
# matmul_3 => clone_15
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7p/c7psq77wycw2ipenowjxpb72atel6k2imrel23m3cystdoecfenm.py
# Source Nodes: [x_15], Original ATen: [aten.view]
# x_15 => view_41
triton_poi_fused_view_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 197)) + (12608*(x0 // 64)) + (75648*(x1 // 197)) + (x0 % 64)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqob7m2omkkerrg4xxhb3i3r4rersjqwnhupxzao3n3g4venmg3.py
# Source Nodes: [l__mod___blocks_0_norm_mlp, patch_embed_5, x_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_norm_mlp => add_21, add_22, mul_17, mul_18, rsqrt_6, sub_8, var_mean_6
# patch_embed_5 => add_20
# x_17 => view_43
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (384*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhqw4rtzfajspg3u2vwltvjfypivps4n3noomvli4lcvdrx5sch.py
# Source Nodes: [x_18, x_21], Original ATen: [aten.gelu, aten.view]
# x_18 => add_23, erf_1, mul_19, mul_20, mul_21
# x_21 => view_45
triton_poi_fused_gelu_view_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2420736
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


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdmuyafmgngf7ii3ocdikdingyitoz6hxzg5z64jpw57hj7pcmp.py
# Source Nodes: [patch_embed_5, patch_embed_7], Original ATen: [aten.add]
# patch_embed_5 => add_20
# patch_embed_7 => add_24
triton_poi_fused_add_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_22', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/74/c74egdcxpprv2tqtsjjzou2xtaojue56yoibi7cdvzn2jdwnp7hf.py
# Source Nodes: [l__mod___blocks_1_norm_mlp_in, pixel_embed_4, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_norm_mlp_in => add_28, add_29, clone_24, mul_25, mul_26, rsqrt_8, sub_11, var_mean_8
# pixel_embed_4 => add_27
# x_26 => view_62
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (24*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxldw4fcbdauxy5kwc77vnzg6sjgm3dgnjkel3dqrpb73i37n6r.py
# Source Nodes: [l__mod___blocks_1_norm1_proj, l__mod___blocks_1_proj, l__mod___blocks_2_attn_in_qk, l__mod___blocks_2_norm_in, pixel_embed_4, pixel_embed_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_norm1_proj => add_32, clone_27, mul_30, rsqrt_9, sub_12, var_mean_9
# l__mod___blocks_1_proj => view_67
# l__mod___blocks_2_attn_in_qk => view_88
# l__mod___blocks_2_norm_in => add_43, mul_41
# pixel_embed_4 => add_27
# pixel_embed_6 => add_31
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (24*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (24*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ak/cak55b22qonm7i33mfzdn2qh7qr7ujpmvert4p3zx4kgn65bbsqp.py
# Source Nodes: [l__mod___blocks_11_norm1_proj, l__mod___blocks_11_proj, pixel_embed_34, pixel_embed_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_11_norm1_proj => add_202, clone_177, mul_210, rsqrt_59, sub_82, var_mean_59
# l__mod___blocks_11_proj => view_477
# pixel_embed_34 => add_197
# pixel_embed_36 => add_201
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (24*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtur52awjfrs5ny6v3x753aibojrcl6emhfylnnoizlfno4g3sa.py
# Source Nodes: [patch_embed_49, patch_embed_51, x_221], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# patch_embed_49 => add_207
# patch_embed_51 => add_211
# x_221 => add_212, mul_220, rsqrt_62, sub_86, var_mean_62
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xi/cxiipp3uiy46sdu7bodkkzat4nr3mfievqocoxccrakoyptyivbl.py
# Source Nodes: [x_223], Original ATen: [aten.clone]
# x_223 => clone_184
triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (75648*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352 = args
    args.clear()
    assert_size_stride(primals_1, (1, 24, 4, 4), (384, 16, 4, 1))
    assert_size_stride(primals_2, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_3, (1, 197, 384), (75648, 384, 1))
    assert_size_stride(primals_4, (24, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_8, (384, 384), (384, 1))
    assert_size_stride(primals_9, (384, ), (1, ))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_11, (384, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (48, 24), (24, 1))
    assert_size_stride(primals_15, (24, 24), (24, 1))
    assert_size_stride(primals_16, (24, 24), (24, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (24, ), (1, ))
    assert_size_stride(primals_20, (96, 24), (24, 1))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (24, 96), (96, 1))
    assert_size_stride(primals_23, (24, ), (1, ))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (24, ), (1, ))
    assert_size_stride(primals_26, (384, 384), (384, 1))
    assert_size_stride(primals_27, (384, ), (1, ))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (768, 384), (384, 1))
    assert_size_stride(primals_31, (384, 384), (384, 1))
    assert_size_stride(primals_32, (384, 384), (384, 1))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (384, ), (1, ))
    assert_size_stride(primals_36, (1536, 384), (384, 1))
    assert_size_stride(primals_37, (1536, ), (1, ))
    assert_size_stride(primals_38, (384, 1536), (1536, 1))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (24, ), (1, ))
    assert_size_stride(primals_41, (24, ), (1, ))
    assert_size_stride(primals_42, (48, 24), (24, 1))
    assert_size_stride(primals_43, (24, 24), (24, 1))
    assert_size_stride(primals_44, (24, 24), (24, 1))
    assert_size_stride(primals_45, (24, ), (1, ))
    assert_size_stride(primals_46, (24, ), (1, ))
    assert_size_stride(primals_47, (24, ), (1, ))
    assert_size_stride(primals_48, (96, 24), (24, 1))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_50, (24, 96), (96, 1))
    assert_size_stride(primals_51, (24, ), (1, ))
    assert_size_stride(primals_52, (24, ), (1, ))
    assert_size_stride(primals_53, (24, ), (1, ))
    assert_size_stride(primals_54, (384, 384), (384, 1))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_58, (768, 384), (384, 1))
    assert_size_stride(primals_59, (384, 384), (384, 1))
    assert_size_stride(primals_60, (384, 384), (384, 1))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (1536, 384), (384, 1))
    assert_size_stride(primals_65, (1536, ), (1, ))
    assert_size_stride(primals_66, (384, 1536), (1536, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (24, ), (1, ))
    assert_size_stride(primals_69, (24, ), (1, ))
    assert_size_stride(primals_70, (48, 24), (24, 1))
    assert_size_stride(primals_71, (24, 24), (24, 1))
    assert_size_stride(primals_72, (24, 24), (24, 1))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_74, (24, ), (1, ))
    assert_size_stride(primals_75, (24, ), (1, ))
    assert_size_stride(primals_76, (96, 24), (24, 1))
    assert_size_stride(primals_77, (96, ), (1, ))
    assert_size_stride(primals_78, (24, 96), (96, 1))
    assert_size_stride(primals_79, (24, ), (1, ))
    assert_size_stride(primals_80, (24, ), (1, ))
    assert_size_stride(primals_81, (24, ), (1, ))
    assert_size_stride(primals_82, (384, 384), (384, 1))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (768, 384), (384, 1))
    assert_size_stride(primals_87, (384, 384), (384, 1))
    assert_size_stride(primals_88, (384, 384), (384, 1))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_92, (1536, 384), (384, 1))
    assert_size_stride(primals_93, (1536, ), (1, ))
    assert_size_stride(primals_94, (384, 1536), (1536, 1))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (24, ), (1, ))
    assert_size_stride(primals_97, (24, ), (1, ))
    assert_size_stride(primals_98, (48, 24), (24, 1))
    assert_size_stride(primals_99, (24, 24), (24, 1))
    assert_size_stride(primals_100, (24, 24), (24, 1))
    assert_size_stride(primals_101, (24, ), (1, ))
    assert_size_stride(primals_102, (24, ), (1, ))
    assert_size_stride(primals_103, (24, ), (1, ))
    assert_size_stride(primals_104, (96, 24), (24, 1))
    assert_size_stride(primals_105, (96, ), (1, ))
    assert_size_stride(primals_106, (24, 96), (96, 1))
    assert_size_stride(primals_107, (24, ), (1, ))
    assert_size_stride(primals_108, (24, ), (1, ))
    assert_size_stride(primals_109, (24, ), (1, ))
    assert_size_stride(primals_110, (384, 384), (384, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_114, (768, 384), (384, 1))
    assert_size_stride(primals_115, (384, 384), (384, 1))
    assert_size_stride(primals_116, (384, 384), (384, 1))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (384, ), (1, ))
    assert_size_stride(primals_120, (1536, 384), (384, 1))
    assert_size_stride(primals_121, (1536, ), (1, ))
    assert_size_stride(primals_122, (384, 1536), (1536, 1))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (24, ), (1, ))
    assert_size_stride(primals_125, (24, ), (1, ))
    assert_size_stride(primals_126, (48, 24), (24, 1))
    assert_size_stride(primals_127, (24, 24), (24, 1))
    assert_size_stride(primals_128, (24, 24), (24, 1))
    assert_size_stride(primals_129, (24, ), (1, ))
    assert_size_stride(primals_130, (24, ), (1, ))
    assert_size_stride(primals_131, (24, ), (1, ))
    assert_size_stride(primals_132, (96, 24), (24, 1))
    assert_size_stride(primals_133, (96, ), (1, ))
    assert_size_stride(primals_134, (24, 96), (96, 1))
    assert_size_stride(primals_135, (24, ), (1, ))
    assert_size_stride(primals_136, (24, ), (1, ))
    assert_size_stride(primals_137, (24, ), (1, ))
    assert_size_stride(primals_138, (384, 384), (384, 1))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (768, 384), (384, 1))
    assert_size_stride(primals_143, (384, 384), (384, 1))
    assert_size_stride(primals_144, (384, 384), (384, 1))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (1536, 384), (384, 1))
    assert_size_stride(primals_149, (1536, ), (1, ))
    assert_size_stride(primals_150, (384, 1536), (1536, 1))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (24, ), (1, ))
    assert_size_stride(primals_153, (24, ), (1, ))
    assert_size_stride(primals_154, (48, 24), (24, 1))
    assert_size_stride(primals_155, (24, 24), (24, 1))
    assert_size_stride(primals_156, (24, 24), (24, 1))
    assert_size_stride(primals_157, (24, ), (1, ))
    assert_size_stride(primals_158, (24, ), (1, ))
    assert_size_stride(primals_159, (24, ), (1, ))
    assert_size_stride(primals_160, (96, 24), (24, 1))
    assert_size_stride(primals_161, (96, ), (1, ))
    assert_size_stride(primals_162, (24, 96), (96, 1))
    assert_size_stride(primals_163, (24, ), (1, ))
    assert_size_stride(primals_164, (24, ), (1, ))
    assert_size_stride(primals_165, (24, ), (1, ))
    assert_size_stride(primals_166, (384, 384), (384, 1))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (768, 384), (384, 1))
    assert_size_stride(primals_171, (384, 384), (384, 1))
    assert_size_stride(primals_172, (384, 384), (384, 1))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (1536, 384), (384, 1))
    assert_size_stride(primals_177, (1536, ), (1, ))
    assert_size_stride(primals_178, (384, 1536), (1536, 1))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (24, ), (1, ))
    assert_size_stride(primals_181, (24, ), (1, ))
    assert_size_stride(primals_182, (48, 24), (24, 1))
    assert_size_stride(primals_183, (24, 24), (24, 1))
    assert_size_stride(primals_184, (24, 24), (24, 1))
    assert_size_stride(primals_185, (24, ), (1, ))
    assert_size_stride(primals_186, (24, ), (1, ))
    assert_size_stride(primals_187, (24, ), (1, ))
    assert_size_stride(primals_188, (96, 24), (24, 1))
    assert_size_stride(primals_189, (96, ), (1, ))
    assert_size_stride(primals_190, (24, 96), (96, 1))
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (24, ), (1, ))
    assert_size_stride(primals_194, (384, 384), (384, 1))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (768, 384), (384, 1))
    assert_size_stride(primals_199, (384, 384), (384, 1))
    assert_size_stride(primals_200, (384, 384), (384, 1))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_203, (384, ), (1, ))
    assert_size_stride(primals_204, (1536, 384), (384, 1))
    assert_size_stride(primals_205, (1536, ), (1, ))
    assert_size_stride(primals_206, (384, 1536), (1536, 1))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_208, (24, ), (1, ))
    assert_size_stride(primals_209, (24, ), (1, ))
    assert_size_stride(primals_210, (48, 24), (24, 1))
    assert_size_stride(primals_211, (24, 24), (24, 1))
    assert_size_stride(primals_212, (24, 24), (24, 1))
    assert_size_stride(primals_213, (24, ), (1, ))
    assert_size_stride(primals_214, (24, ), (1, ))
    assert_size_stride(primals_215, (24, ), (1, ))
    assert_size_stride(primals_216, (96, 24), (24, 1))
    assert_size_stride(primals_217, (96, ), (1, ))
    assert_size_stride(primals_218, (24, 96), (96, 1))
    assert_size_stride(primals_219, (24, ), (1, ))
    assert_size_stride(primals_220, (24, ), (1, ))
    assert_size_stride(primals_221, (24, ), (1, ))
    assert_size_stride(primals_222, (384, 384), (384, 1))
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_224, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (768, 384), (384, 1))
    assert_size_stride(primals_227, (384, 384), (384, 1))
    assert_size_stride(primals_228, (384, 384), (384, 1))
    assert_size_stride(primals_229, (384, ), (1, ))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (1536, 384), (384, 1))
    assert_size_stride(primals_233, (1536, ), (1, ))
    assert_size_stride(primals_234, (384, 1536), (1536, 1))
    assert_size_stride(primals_235, (384, ), (1, ))
    assert_size_stride(primals_236, (24, ), (1, ))
    assert_size_stride(primals_237, (24, ), (1, ))
    assert_size_stride(primals_238, (48, 24), (24, 1))
    assert_size_stride(primals_239, (24, 24), (24, 1))
    assert_size_stride(primals_240, (24, 24), (24, 1))
    assert_size_stride(primals_241, (24, ), (1, ))
    assert_size_stride(primals_242, (24, ), (1, ))
    assert_size_stride(primals_243, (24, ), (1, ))
    assert_size_stride(primals_244, (96, 24), (24, 1))
    assert_size_stride(primals_245, (96, ), (1, ))
    assert_size_stride(primals_246, (24, 96), (96, 1))
    assert_size_stride(primals_247, (24, ), (1, ))
    assert_size_stride(primals_248, (24, ), (1, ))
    assert_size_stride(primals_249, (24, ), (1, ))
    assert_size_stride(primals_250, (384, 384), (384, 1))
    assert_size_stride(primals_251, (384, ), (1, ))
    assert_size_stride(primals_252, (384, ), (1, ))
    assert_size_stride(primals_253, (384, ), (1, ))
    assert_size_stride(primals_254, (768, 384), (384, 1))
    assert_size_stride(primals_255, (384, 384), (384, 1))
    assert_size_stride(primals_256, (384, 384), (384, 1))
    assert_size_stride(primals_257, (384, ), (1, ))
    assert_size_stride(primals_258, (384, ), (1, ))
    assert_size_stride(primals_259, (384, ), (1, ))
    assert_size_stride(primals_260, (1536, 384), (384, 1))
    assert_size_stride(primals_261, (1536, ), (1, ))
    assert_size_stride(primals_262, (384, 1536), (1536, 1))
    assert_size_stride(primals_263, (384, ), (1, ))
    assert_size_stride(primals_264, (24, ), (1, ))
    assert_size_stride(primals_265, (24, ), (1, ))
    assert_size_stride(primals_266, (48, 24), (24, 1))
    assert_size_stride(primals_267, (24, 24), (24, 1))
    assert_size_stride(primals_268, (24, 24), (24, 1))
    assert_size_stride(primals_269, (24, ), (1, ))
    assert_size_stride(primals_270, (24, ), (1, ))
    assert_size_stride(primals_271, (24, ), (1, ))
    assert_size_stride(primals_272, (96, 24), (24, 1))
    assert_size_stride(primals_273, (96, ), (1, ))
    assert_size_stride(primals_274, (24, 96), (96, 1))
    assert_size_stride(primals_275, (24, ), (1, ))
    assert_size_stride(primals_276, (24, ), (1, ))
    assert_size_stride(primals_277, (24, ), (1, ))
    assert_size_stride(primals_278, (384, 384), (384, 1))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_281, (384, ), (1, ))
    assert_size_stride(primals_282, (768, 384), (384, 1))
    assert_size_stride(primals_283, (384, 384), (384, 1))
    assert_size_stride(primals_284, (384, 384), (384, 1))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_287, (384, ), (1, ))
    assert_size_stride(primals_288, (1536, 384), (384, 1))
    assert_size_stride(primals_289, (1536, ), (1, ))
    assert_size_stride(primals_290, (384, 1536), (1536, 1))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_292, (24, ), (1, ))
    assert_size_stride(primals_293, (24, ), (1, ))
    assert_size_stride(primals_294, (48, 24), (24, 1))
    assert_size_stride(primals_295, (24, 24), (24, 1))
    assert_size_stride(primals_296, (24, 24), (24, 1))
    assert_size_stride(primals_297, (24, ), (1, ))
    assert_size_stride(primals_298, (24, ), (1, ))
    assert_size_stride(primals_299, (24, ), (1, ))
    assert_size_stride(primals_300, (96, 24), (24, 1))
    assert_size_stride(primals_301, (96, ), (1, ))
    assert_size_stride(primals_302, (24, 96), (96, 1))
    assert_size_stride(primals_303, (24, ), (1, ))
    assert_size_stride(primals_304, (24, ), (1, ))
    assert_size_stride(primals_305, (24, ), (1, ))
    assert_size_stride(primals_306, (384, 384), (384, 1))
    assert_size_stride(primals_307, (384, ), (1, ))
    assert_size_stride(primals_308, (384, ), (1, ))
    assert_size_stride(primals_309, (384, ), (1, ))
    assert_size_stride(primals_310, (768, 384), (384, 1))
    assert_size_stride(primals_311, (384, 384), (384, 1))
    assert_size_stride(primals_312, (384, 384), (384, 1))
    assert_size_stride(primals_313, (384, ), (1, ))
    assert_size_stride(primals_314, (384, ), (1, ))
    assert_size_stride(primals_315, (384, ), (1, ))
    assert_size_stride(primals_316, (1536, 384), (384, 1))
    assert_size_stride(primals_317, (1536, ), (1, ))
    assert_size_stride(primals_318, (384, 1536), (1536, 1))
    assert_size_stride(primals_319, (384, ), (1, ))
    assert_size_stride(primals_320, (24, ), (1, ))
    assert_size_stride(primals_321, (24, ), (1, ))
    assert_size_stride(primals_322, (48, 24), (24, 1))
    assert_size_stride(primals_323, (24, 24), (24, 1))
    assert_size_stride(primals_324, (24, 24), (24, 1))
    assert_size_stride(primals_325, (24, ), (1, ))
    assert_size_stride(primals_326, (24, ), (1, ))
    assert_size_stride(primals_327, (24, ), (1, ))
    assert_size_stride(primals_328, (96, 24), (24, 1))
    assert_size_stride(primals_329, (96, ), (1, ))
    assert_size_stride(primals_330, (24, 96), (96, 1))
    assert_size_stride(primals_331, (24, ), (1, ))
    assert_size_stride(primals_332, (24, ), (1, ))
    assert_size_stride(primals_333, (24, ), (1, ))
    assert_size_stride(primals_334, (384, 384), (384, 1))
    assert_size_stride(primals_335, (384, ), (1, ))
    assert_size_stride(primals_336, (384, ), (1, ))
    assert_size_stride(primals_337, (384, ), (1, ))
    assert_size_stride(primals_338, (768, 384), (384, 1))
    assert_size_stride(primals_339, (384, 384), (384, 1))
    assert_size_stride(primals_340, (384, 384), (384, 1))
    assert_size_stride(primals_341, (384, ), (1, ))
    assert_size_stride(primals_342, (384, ), (1, ))
    assert_size_stride(primals_343, (384, ), (1, ))
    assert_size_stride(primals_344, (1536, 384), (384, 1))
    assert_size_stride(primals_345, (1536, ), (1, ))
    assert_size_stride(primals_346, (384, 1536), (1536, 1))
    assert_size_stride(primals_347, (384, ), (1, ))
    assert_size_stride(primals_348, (384, ), (1, ))
    assert_size_stride(primals_349, (384, ), (1, ))
    assert_size_stride(primals_350, (1000, 384), (384, 1))
    assert_size_stride(primals_351, (1000, ), (1, ))
    assert_size_stride(primals_352, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_352, primals_4, stride=(4, 4), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf1 = empty((4, 14), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_1], Original ATen: [aten.im2col]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_im2col_0.run(buf1, 56, grid=grid(56), stream=stream0)
        buf2 = empty((1568, 24, 4, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_3], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf1, buf0, primals_5, primals_1, buf2, 602112, grid=grid(602112), stream=stream0)
        del primals_1
        del primals_5
        buf3 = reinterpret_tensor(buf0, (1568, 16, 24), (384, 24, 1), 0); del buf0  # reuse
        buf15 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cuda', dtype=torch.float32)
        buf18 = reinterpret_tensor(buf16, (1568, 16, 1), (16, 1, 1), 0); del buf16  # reuse
        # Source Nodes: [l__mod___blocks_0_norm_in, reshape_2], Original ATen: [aten.clone, aten.native_layer_norm]
        triton_per_fused_clone_native_layer_norm_2.run(buf18, buf2, buf3, buf15, 25088, 24, grid=grid(25088), stream=stream0)
        buf4 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf7 = reinterpret_tensor(buf5, (8, 196, 1), (196, 1, 1), 0); del buf5  # reuse
        buf8 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf19 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_in_qk, l__mod___blocks_0_norm_in, l__mod___norm1_proj, l__mod___proj], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_3.run(buf7, buf3, primals_6, primals_7, buf15, buf18, primals_12, primals_13, buf4, buf8, buf19, 1568, 384, grid=grid(1568), stream=stream0)
        del primals_13
        del primals_7
        buf9 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___proj], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf8, reinterpret_tensor(primals_8, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf9)
        del primals_9
        buf10 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf13 = reinterpret_tensor(buf11, (8, 196, 1), (196, 1, 1), 0); del buf11  # reuse
        # Source Nodes: [patch_embed], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_4.run(buf13, buf9, buf10, 1568, 384, grid=grid(1568), stream=stream0)
        buf14 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25, patch_embed_2], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_5.run(primals_2, buf9, buf10, buf13, primals_10, primals_11, primals_3, buf14, 605184, grid=grid(605184), stream=stream0)
        del primals_11
        del primals_2
        del primals_3
        buf20 = empty((25088, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_14, (24, 48), (1, 24), 0), out=buf20)
        buf21 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_15, (24, 24), (1, 24), 0), out=buf21)
        buf22 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf20, buf22, 602112, grid=grid(602112), stream=stream0)
        buf23 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf20, buf23, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf24 = empty((6272, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf22, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf23, (6272, 6, 16), (96, 16, 1), 0), out=buf24)
        buf27 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf24, buf27, 100352, 16, grid=grid(100352), stream=stream0)
        buf28 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf21, buf28, 602112, grid=grid(602112), stream=stream0)
        buf29 = reinterpret_tensor(buf21, (6272, 16, 6), (96, 6, 1), 0); del buf21  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf28, (6272, 16, 6), (96, 6, 1), 0), out=buf29)
        buf30 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf29, buf30, 602112, grid=grid(602112), stream=stream0)
        buf31 = reinterpret_tensor(buf29, (25088, 24), (24, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf30, reinterpret_tensor(primals_16, (24, 24), (1, 24), 0), out=buf31)
        buf35 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf36 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf721 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm_mlp_in, pixel_embed_1, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf2, buf31, primals_17, primals_18, primals_19, buf35, buf36, buf721, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_19
        buf37 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, buf36, reinterpret_tensor(primals_20, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf37)
        del primals_21
        buf38 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf37, buf38, 2408448, grid=grid(2408448), stream=stream0)
        buf39 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf38, reinterpret_tensor(primals_22, (96, 24), (1, 96), 0), out=buf39)
        buf40 = reinterpret_tensor(buf39, (1568, 16, 24), (384, 24, 1), 0); del buf39  # reuse
        buf44 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf45 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf74 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf719 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm1_proj, l__mod___blocks_0_proj, l__mod___blocks_1_attn_in_qk, l__mod___blocks_1_norm_in, pixel_embed_1, pixel_embed_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf40, buf2, buf31, primals_17, primals_23, primals_24, primals_25, primals_40, primals_41, buf44, buf45, buf74, buf719, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_17
        del primals_23
        del primals_25
        del primals_41
        buf46 = reinterpret_tensor(buf31, (1568, 384), (384, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf45, reinterpret_tensor(primals_26, (384, 384), (1, 384), 0), out=buf46)
        buf47 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf48 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf51 = reinterpret_tensor(buf49, (8, 197, 1), (197, 1, 1), 0); del buf49  # reuse
        buf52 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24, l__mod___blocks_0_attn_out_qk, l__mod___blocks_0_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf51, buf14, buf46, primals_27, primals_28, primals_29, buf47, buf48, buf52, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_27
        del primals_29
        buf53 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_30, (384, 768), (1, 384), 0), out=buf53)
        buf54 = reinterpret_tensor(buf14, (1576, 384), (384, 1), 0); del buf14  # reuse
        # Source Nodes: [l__mod___blocks_0_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_31, (384, 384), (1, 384), 0), out=buf54)
        buf55 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf53, buf55, 605184, grid=grid(605184), stream=stream0)
        buf56 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf53, buf56, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf57 = empty((48, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf56, (48, 64, 197), (12608, 197, 1), 0), out=buf57)
        buf60 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf57, buf60, 9456, 197, grid=grid(9456), stream=stream0)
        buf61 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf54, buf61, 605184, grid=grid(605184), stream=stream0)
        buf62 = reinterpret_tensor(buf54, (48, 197, 64), (12608, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf61, (48, 197, 64), (12608, 64, 1), 0), out=buf62)
        buf63 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf62, buf63, 605184, grid=grid(605184), stream=stream0)
        buf64 = reinterpret_tensor(buf62, (1576, 384), (384, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf63, reinterpret_tensor(primals_32, (384, 384), (1, 384), 0), out=buf64)
        buf68 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf69 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf720 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_norm_mlp, patch_embed_5, x_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf47, buf64, primals_33, primals_34, primals_35, buf68, buf69, buf720, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_35
        buf70 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, buf69, reinterpret_tensor(primals_36, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf70)
        del primals_37
        buf71 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18, x_21], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf70, buf71, 2420736, grid=grid(2420736), stream=stream0)
        buf72 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf71, reinterpret_tensor(primals_38, (1536, 384), (1, 1536), 0), out=buf72)
        buf73 = reinterpret_tensor(buf72, (8, 197, 384), (75648, 384, 1), 0); del buf72  # reuse
        # Source Nodes: [patch_embed_5, patch_embed_7], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf73, buf47, buf64, primals_33, primals_39, 605184, grid=grid(605184), stream=stream0)
        del primals_33
        del primals_39
        buf75 = buf20; del buf20  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_42, (24, 48), (1, 24), 0), out=buf75)
        buf76 = reinterpret_tensor(buf46, (25088, 24), (24, 1), 0); del buf46  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_43, (24, 24), (1, 24), 0), out=buf76)
        buf77 = reinterpret_tensor(buf2, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf2  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf75, buf77, 602112, grid=grid(602112), stream=stream0)
        buf78 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf75, buf78, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf79 = buf24; del buf24  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf78, (6272, 6, 16), (96, 16, 1), 0), out=buf79)
        buf82 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf79, buf82, 100352, 16, grid=grid(100352), stream=stream0)
        buf83 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf76, buf83, 602112, grid=grid(602112), stream=stream0)
        buf84 = reinterpret_tensor(buf76, (6272, 16, 6), (96, 6, 1), 0); del buf76  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf83, (6272, 16, 6), (96, 6, 1), 0), out=buf84)
        buf85 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf84, buf85, 602112, grid=grid(602112), stream=stream0)
        buf86 = reinterpret_tensor(buf84, (25088, 24), (24, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf85, reinterpret_tensor(primals_44, (24, 24), (1, 24), 0), out=buf86)
        buf90 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf91 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf718 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_norm_mlp_in, pixel_embed_4, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf40, buf86, primals_45, primals_46, primals_47, buf90, buf91, buf718, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_47
        buf92 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf91, reinterpret_tensor(primals_48, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf92)
        del primals_49
        buf93 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27, x_30], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf92, buf93, 2408448, grid=grid(2408448), stream=stream0)
        buf94 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf93, reinterpret_tensor(primals_50, (96, 24), (1, 96), 0), out=buf94)
        buf95 = reinterpret_tensor(buf94, (1568, 16, 24), (384, 24, 1), 0); del buf94  # reuse
        buf99 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf100 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf129 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf716 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_norm1_proj, l__mod___blocks_1_proj, l__mod___blocks_2_attn_in_qk, l__mod___blocks_2_norm_in, pixel_embed_4, pixel_embed_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf95, buf40, buf86, primals_45, primals_51, primals_52, primals_53, primals_68, primals_69, buf99, buf100, buf129, buf716, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_45
        del primals_51
        del primals_53
        del primals_69
        buf101 = reinterpret_tensor(buf86, (1568, 384), (384, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf100, reinterpret_tensor(primals_54, (384, 384), (1, 384), 0), out=buf101)
        buf102 = reinterpret_tensor(buf64, (8, 197, 384), (75648, 384, 1), 0); del buf64  # reuse
        buf103 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf106 = reinterpret_tensor(buf104, (8, 197, 1), (197, 1, 1), 0); del buf104  # reuse
        buf107 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23, l__mod___blocks_1_attn_out_qk, l__mod___blocks_1_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf106, buf73, buf101, primals_55, primals_56, primals_57, buf102, buf103, buf107, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_55
        del primals_57
        buf108 = buf53; del buf53  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf107, reinterpret_tensor(primals_58, (384, 768), (1, 384), 0), out=buf108)
        buf109 = reinterpret_tensor(buf73, (1576, 384), (384, 1), 0); del buf73  # reuse
        # Source Nodes: [l__mod___blocks_1_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf107, reinterpret_tensor(primals_59, (384, 384), (1, 384), 0), out=buf109)
        buf110 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf108, buf110, 605184, grid=grid(605184), stream=stream0)
        buf111 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf108, buf111, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf112 = buf57; del buf57  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf111, (48, 64, 197), (12608, 197, 1), 0), out=buf112)
        buf115 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, attn_9], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf112, buf115, 9456, 197, grid=grid(9456), stream=stream0)
        buf116 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf109, buf116, 605184, grid=grid(605184), stream=stream0)
        buf117 = reinterpret_tensor(buf109, (48, 197, 64), (12608, 64, 1), 0); del buf109  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf116, (48, 197, 64), (12608, 64, 1), 0), out=buf117)
        buf118 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf117, buf118, 605184, grid=grid(605184), stream=stream0)
        buf119 = reinterpret_tensor(buf117, (1576, 384), (384, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf118, reinterpret_tensor(primals_60, (384, 384), (1, 384), 0), out=buf119)
        buf123 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf124 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf717 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_norm_mlp, patch_embed_9, x_35], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf102, buf119, primals_61, primals_62, primals_63, buf123, buf124, buf717, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_63
        buf125 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf124, reinterpret_tensor(primals_64, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf125)
        del primals_65
        buf126 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36, x_39], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf125, buf126, 2420736, grid=grid(2420736), stream=stream0)
        buf127 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf126, reinterpret_tensor(primals_66, (1536, 384), (1, 1536), 0), out=buf127)
        buf128 = reinterpret_tensor(buf127, (8, 197, 384), (75648, 384, 1), 0); del buf127  # reuse
        # Source Nodes: [patch_embed_11, patch_embed_9], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf128, buf102, buf119, primals_61, primals_67, 605184, grid=grid(605184), stream=stream0)
        del primals_61
        del primals_67
        buf130 = buf75; del buf75  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf129, reinterpret_tensor(primals_70, (24, 48), (1, 24), 0), out=buf130)
        buf131 = reinterpret_tensor(buf101, (25088, 24), (24, 1), 0); del buf101  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf129, reinterpret_tensor(primals_71, (24, 24), (1, 24), 0), out=buf131)
        buf132 = reinterpret_tensor(buf40, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf40  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf130, buf132, 602112, grid=grid(602112), stream=stream0)
        buf133 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf130, buf133, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf134 = buf79; del buf79  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf133, (6272, 6, 16), (96, 16, 1), 0), out=buf134)
        buf137 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf134, buf137, 100352, 16, grid=grid(100352), stream=stream0)
        buf138 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf131, buf138, 602112, grid=grid(602112), stream=stream0)
        buf139 = reinterpret_tensor(buf131, (6272, 16, 6), (96, 6, 1), 0); del buf131  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf138, (6272, 16, 6), (96, 6, 1), 0), out=buf139)
        buf140 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_42], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf139, buf140, 602112, grid=grid(602112), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (25088, 24), (24, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf140, reinterpret_tensor(primals_72, (24, 24), (1, 24), 0), out=buf141)
        buf145 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf146 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf715 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_norm_mlp_in, pixel_embed_7, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf95, buf141, primals_73, primals_74, primals_75, buf145, buf146, buf715, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_75
        buf147 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, buf146, reinterpret_tensor(primals_76, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf147)
        del primals_77
        buf148 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_48], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf147, buf148, 2408448, grid=grid(2408448), stream=stream0)
        buf149 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf148, reinterpret_tensor(primals_78, (96, 24), (1, 96), 0), out=buf149)
        buf150 = reinterpret_tensor(buf149, (1568, 16, 24), (384, 24, 1), 0); del buf149  # reuse
        buf154 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf155 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf184 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf713 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_norm1_proj, l__mod___blocks_2_proj, l__mod___blocks_3_attn_in_qk, l__mod___blocks_3_norm_in, pixel_embed_7, pixel_embed_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf150, buf95, buf141, primals_73, primals_79, primals_80, primals_81, primals_96, primals_97, buf154, buf155, buf184, buf713, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_73
        del primals_79
        del primals_81
        del primals_97
        buf156 = reinterpret_tensor(buf95, (1568, 384), (384, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf155, reinterpret_tensor(primals_82, (384, 384), (1, 384), 0), out=buf156)
        buf157 = reinterpret_tensor(buf119, (8, 197, 384), (75648, 384, 1), 0); del buf119  # reuse
        buf158 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf159 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf161 = reinterpret_tensor(buf159, (8, 197, 1), (197, 1, 1), 0); del buf159  # reuse
        buf162 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_22, l__mod___blocks_2_attn_out_qk, l__mod___blocks_2_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf161, buf128, buf156, primals_83, primals_84, primals_85, buf157, buf158, buf162, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_83
        del primals_85
        buf163 = buf108; del buf108  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf162, reinterpret_tensor(primals_86, (384, 768), (1, 384), 0), out=buf163)
        buf164 = reinterpret_tensor(buf128, (1576, 384), (384, 1), 0); del buf128  # reuse
        # Source Nodes: [l__mod___blocks_2_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf162, reinterpret_tensor(primals_87, (384, 384), (1, 384), 0), out=buf164)
        buf165 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf163, buf165, 605184, grid=grid(605184), stream=stream0)
        buf166 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf163, buf166, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf167 = buf112; del buf112  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf165, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf166, (48, 64, 197), (12608, 197, 1), 0), out=buf167)
        buf170 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_15, attn_16], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf167, buf170, 9456, 197, grid=grid(9456), stream=stream0)
        buf171 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf164, buf171, 605184, grid=grid(605184), stream=stream0)
        buf172 = reinterpret_tensor(buf164, (48, 197, 64), (12608, 64, 1), 0); del buf164  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf170, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf171, (48, 197, 64), (12608, 64, 1), 0), out=buf172)
        buf173 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf172, buf173, 605184, grid=grid(605184), stream=stream0)
        buf174 = reinterpret_tensor(buf172, (1576, 384), (384, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf173, reinterpret_tensor(primals_88, (384, 384), (1, 384), 0), out=buf174)
        buf178 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf179 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf714 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_norm_mlp, patch_embed_13, x_53], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf157, buf174, primals_89, primals_90, primals_91, buf178, buf179, buf714, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_91
        buf180 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, buf179, reinterpret_tensor(primals_92, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf180)
        del primals_93
        buf181 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54, x_57], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf180, buf181, 2420736, grid=grid(2420736), stream=stream0)
        buf182 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf181, reinterpret_tensor(primals_94, (1536, 384), (1, 1536), 0), out=buf182)
        buf183 = reinterpret_tensor(buf182, (8, 197, 384), (75648, 384, 1), 0); del buf182  # reuse
        # Source Nodes: [patch_embed_13, patch_embed_15], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf183, buf157, buf174, primals_89, primals_95, 605184, grid=grid(605184), stream=stream0)
        del primals_89
        del primals_95
        buf185 = buf130; del buf130  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf184, reinterpret_tensor(primals_98, (24, 48), (1, 24), 0), out=buf185)
        buf186 = reinterpret_tensor(buf156, (25088, 24), (24, 1), 0); del buf156  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf184, reinterpret_tensor(primals_99, (24, 24), (1, 24), 0), out=buf186)
        buf187 = reinterpret_tensor(buf141, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf141  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf185, buf187, 602112, grid=grid(602112), stream=stream0)
        buf188 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf185, buf188, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf189 = buf134; del buf134  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf188, (6272, 6, 16), (96, 16, 1), 0), out=buf189)
        buf192 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_18, attn_19], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf189, buf192, 100352, 16, grid=grid(100352), stream=stream0)
        buf193 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf186, buf193, 602112, grid=grid(602112), stream=stream0)
        buf194 = reinterpret_tensor(buf186, (6272, 16, 6), (96, 6, 1), 0); del buf186  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf192, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf193, (6272, 16, 6), (96, 6, 1), 0), out=buf194)
        buf195 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf194, buf195, 602112, grid=grid(602112), stream=stream0)
        buf196 = reinterpret_tensor(buf194, (25088, 24), (24, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf195, reinterpret_tensor(primals_100, (24, 24), (1, 24), 0), out=buf196)
        buf200 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf201 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf712 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_norm_mlp_in, pixel_embed_10, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf150, buf196, primals_101, primals_102, primals_103, buf200, buf201, buf712, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_103
        buf202 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_105, buf201, reinterpret_tensor(primals_104, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf202)
        del primals_105
        buf203 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_66], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf202, buf203, 2408448, grid=grid(2408448), stream=stream0)
        buf204 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf203, reinterpret_tensor(primals_106, (96, 24), (1, 96), 0), out=buf204)
        buf205 = reinterpret_tensor(buf204, (1568, 16, 24), (384, 24, 1), 0); del buf204  # reuse
        buf209 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf210 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf239 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf710 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_norm1_proj, l__mod___blocks_3_proj, l__mod___blocks_4_attn_in_qk, l__mod___blocks_4_norm_in, pixel_embed_10, pixel_embed_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf205, buf150, buf196, primals_101, primals_107, primals_108, primals_109, primals_124, primals_125, buf209, buf210, buf239, buf710, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_101
        del primals_107
        del primals_109
        del primals_125
        buf211 = reinterpret_tensor(buf196, (1568, 384), (384, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf210, reinterpret_tensor(primals_110, (384, 384), (1, 384), 0), out=buf211)
        buf212 = reinterpret_tensor(buf174, (8, 197, 384), (75648, 384, 1), 0); del buf174  # reuse
        buf213 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf214 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf216 = reinterpret_tensor(buf214, (8, 197, 1), (197, 1, 1), 0); del buf214  # reuse
        buf217 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_21, l__mod___blocks_3_attn_out_qk, l__mod___blocks_3_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf216, buf183, buf211, primals_111, primals_112, primals_113, buf212, buf213, buf217, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_111
        del primals_113
        buf218 = buf163; del buf163  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf217, reinterpret_tensor(primals_114, (384, 768), (1, 384), 0), out=buf218)
        buf219 = reinterpret_tensor(buf183, (1576, 384), (384, 1), 0); del buf183  # reuse
        # Source Nodes: [l__mod___blocks_3_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf217, reinterpret_tensor(primals_115, (384, 384), (1, 384), 0), out=buf219)
        buf220 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf218, buf220, 605184, grid=grid(605184), stream=stream0)
        buf221 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf218, buf221, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf222 = buf167; del buf167  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf221, (48, 64, 197), (12608, 197, 1), 0), out=buf222)
        buf225 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_21, attn_22], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf222, buf225, 9456, 197, grid=grid(9456), stream=stream0)
        buf226 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf219, buf226, 605184, grid=grid(605184), stream=stream0)
        buf227 = reinterpret_tensor(buf219, (48, 197, 64), (12608, 64, 1), 0); del buf219  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf226, (48, 197, 64), (12608, 64, 1), 0), out=buf227)
        buf228 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf227, buf228, 605184, grid=grid(605184), stream=stream0)
        buf229 = reinterpret_tensor(buf227, (1576, 384), (384, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf228, reinterpret_tensor(primals_116, (384, 384), (1, 384), 0), out=buf229)
        buf233 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf234 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf711 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_3_norm_mlp, patch_embed_17, x_71], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf212, buf229, primals_117, primals_118, primals_119, buf233, buf234, buf711, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_119
        buf235 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_121, buf234, reinterpret_tensor(primals_120, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf235)
        del primals_121
        buf236 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72, x_75], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf235, buf236, 2420736, grid=grid(2420736), stream=stream0)
        buf237 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_122, (1536, 384), (1, 1536), 0), out=buf237)
        buf238 = reinterpret_tensor(buf237, (8, 197, 384), (75648, 384, 1), 0); del buf237  # reuse
        # Source Nodes: [patch_embed_17, patch_embed_19], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf238, buf212, buf229, primals_117, primals_123, 605184, grid=grid(605184), stream=stream0)
        del primals_117
        del primals_123
        buf240 = buf185; del buf185  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf239, reinterpret_tensor(primals_126, (24, 48), (1, 24), 0), out=buf240)
        buf241 = reinterpret_tensor(buf211, (25088, 24), (24, 1), 0); del buf211  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf239, reinterpret_tensor(primals_127, (24, 24), (1, 24), 0), out=buf241)
        buf242 = reinterpret_tensor(buf150, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf150  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf240, buf242, 602112, grid=grid(602112), stream=stream0)
        buf243 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf240, buf243, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf244 = buf189; del buf189  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf243, (6272, 6, 16), (96, 16, 1), 0), out=buf244)
        buf247 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_24, attn_25], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf244, buf247, 100352, 16, grid=grid(100352), stream=stream0)
        buf248 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf241, buf248, 602112, grid=grid(602112), stream=stream0)
        buf249 = reinterpret_tensor(buf241, (6272, 16, 6), (96, 6, 1), 0); del buf241  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf248, (6272, 16, 6), (96, 6, 1), 0), out=buf249)
        buf250 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf249, buf250, 602112, grid=grid(602112), stream=stream0)
        buf251 = reinterpret_tensor(buf249, (25088, 24), (24, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf250, reinterpret_tensor(primals_128, (24, 24), (1, 24), 0), out=buf251)
        buf255 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf256 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf709 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_norm_mlp_in, pixel_embed_13, x_80], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf205, buf251, primals_129, primals_130, primals_131, buf255, buf256, buf709, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_131
        buf257 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_133, buf256, reinterpret_tensor(primals_132, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf257)
        del primals_133
        buf258 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81, x_84], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf257, buf258, 2408448, grid=grid(2408448), stream=stream0)
        buf259 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf258, reinterpret_tensor(primals_134, (96, 24), (1, 96), 0), out=buf259)
        buf260 = reinterpret_tensor(buf259, (1568, 16, 24), (384, 24, 1), 0); del buf259  # reuse
        buf264 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf265 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf294 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf707 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_norm1_proj, l__mod___blocks_4_proj, l__mod___blocks_5_attn_in_qk, l__mod___blocks_5_norm_in, pixel_embed_13, pixel_embed_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf260, buf205, buf251, primals_129, primals_135, primals_136, primals_137, primals_152, primals_153, buf264, buf265, buf294, buf707, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_129
        del primals_135
        del primals_137
        del primals_153
        buf266 = reinterpret_tensor(buf251, (1568, 384), (384, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf265, reinterpret_tensor(primals_138, (384, 384), (1, 384), 0), out=buf266)
        buf267 = reinterpret_tensor(buf229, (8, 197, 384), (75648, 384, 1), 0); del buf229  # reuse
        buf268 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf269 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf271 = reinterpret_tensor(buf269, (8, 197, 1), (197, 1, 1), 0); del buf269  # reuse
        buf272 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_20, l__mod___blocks_4_attn_out_qk, l__mod___blocks_4_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf271, buf238, buf266, primals_139, primals_140, primals_141, buf267, buf268, buf272, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_139
        del primals_141
        buf273 = buf218; del buf218  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf272, reinterpret_tensor(primals_142, (384, 768), (1, 384), 0), out=buf273)
        buf274 = reinterpret_tensor(buf238, (1576, 384), (384, 1), 0); del buf238  # reuse
        # Source Nodes: [l__mod___blocks_4_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf272, reinterpret_tensor(primals_143, (384, 384), (1, 384), 0), out=buf274)
        buf275 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf273, buf275, 605184, grid=grid(605184), stream=stream0)
        buf276 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf273, buf276, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf277 = buf222; del buf222  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf275, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf276, (48, 64, 197), (12608, 197, 1), 0), out=buf277)
        buf280 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_27, attn_28], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf277, buf280, 9456, 197, grid=grid(9456), stream=stream0)
        buf281 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf274, buf281, 605184, grid=grid(605184), stream=stream0)
        buf282 = reinterpret_tensor(buf274, (48, 197, 64), (12608, 64, 1), 0); del buf274  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf280, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf281, (48, 197, 64), (12608, 64, 1), 0), out=buf282)
        buf283 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf282, buf283, 605184, grid=grid(605184), stream=stream0)
        buf284 = reinterpret_tensor(buf282, (1576, 384), (384, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf283, reinterpret_tensor(primals_144, (384, 384), (1, 384), 0), out=buf284)
        buf288 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf289 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf708 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_4_norm_mlp, patch_embed_21, x_89], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf267, buf284, primals_145, primals_146, primals_147, buf288, buf289, buf708, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_147
        buf290 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_149, buf289, reinterpret_tensor(primals_148, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf290)
        del primals_149
        buf291 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90, x_93], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf290, buf291, 2420736, grid=grid(2420736), stream=stream0)
        buf292 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf291, reinterpret_tensor(primals_150, (1536, 384), (1, 1536), 0), out=buf292)
        buf293 = reinterpret_tensor(buf292, (8, 197, 384), (75648, 384, 1), 0); del buf292  # reuse
        # Source Nodes: [patch_embed_21, patch_embed_23], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf293, buf267, buf284, primals_145, primals_151, 605184, grid=grid(605184), stream=stream0)
        del primals_145
        del primals_151
        buf295 = buf240; del buf240  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf294, reinterpret_tensor(primals_154, (24, 48), (1, 24), 0), out=buf295)
        buf296 = reinterpret_tensor(buf266, (25088, 24), (24, 1), 0); del buf266  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf294, reinterpret_tensor(primals_155, (24, 24), (1, 24), 0), out=buf296)
        buf297 = reinterpret_tensor(buf205, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf205  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf295, buf297, 602112, grid=grid(602112), stream=stream0)
        buf298 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf295, buf298, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf299 = buf244; del buf244  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf297, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf298, (6272, 6, 16), (96, 16, 1), 0), out=buf299)
        buf302 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_30, attn_31], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf299, buf302, 100352, 16, grid=grid(100352), stream=stream0)
        buf303 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf296, buf303, 602112, grid=grid(602112), stream=stream0)
        buf304 = reinterpret_tensor(buf296, (6272, 16, 6), (96, 6, 1), 0); del buf296  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf302, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf303, (6272, 16, 6), (96, 6, 1), 0), out=buf304)
        buf305 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf304, buf305, 602112, grid=grid(602112), stream=stream0)
        buf306 = reinterpret_tensor(buf304, (25088, 24), (24, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf305, reinterpret_tensor(primals_156, (24, 24), (1, 24), 0), out=buf306)
        buf310 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf311 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf706 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_norm_mlp_in, pixel_embed_16, x_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf260, buf306, primals_157, primals_158, primals_159, buf310, buf311, buf706, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_159
        buf312 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf311, reinterpret_tensor(primals_160, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf312)
        del primals_161
        buf313 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_99], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf312, buf313, 2408448, grid=grid(2408448), stream=stream0)
        buf314 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf313, reinterpret_tensor(primals_162, (96, 24), (1, 96), 0), out=buf314)
        buf315 = reinterpret_tensor(buf314, (1568, 16, 24), (384, 24, 1), 0); del buf314  # reuse
        buf319 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf320 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf349 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf704 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_norm1_proj, l__mod___blocks_5_proj, l__mod___blocks_6_attn_in_qk, l__mod___blocks_6_norm_in, pixel_embed_16, pixel_embed_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf315, buf260, buf306, primals_157, primals_163, primals_164, primals_165, primals_180, primals_181, buf319, buf320, buf349, buf704, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_157
        del primals_163
        del primals_165
        del primals_181
        buf321 = reinterpret_tensor(buf306, (1568, 384), (384, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf320, reinterpret_tensor(primals_166, (384, 384), (1, 384), 0), out=buf321)
        buf322 = reinterpret_tensor(buf284, (8, 197, 384), (75648, 384, 1), 0); del buf284  # reuse
        buf323 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf324 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf326 = reinterpret_tensor(buf324, (8, 197, 1), (197, 1, 1), 0); del buf324  # reuse
        buf327 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_19, l__mod___blocks_5_attn_out_qk, l__mod___blocks_5_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf326, buf293, buf321, primals_167, primals_168, primals_169, buf322, buf323, buf327, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_167
        del primals_169
        buf328 = buf273; del buf273  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf327, reinterpret_tensor(primals_170, (384, 768), (1, 384), 0), out=buf328)
        buf329 = reinterpret_tensor(buf293, (1576, 384), (384, 1), 0); del buf293  # reuse
        # Source Nodes: [l__mod___blocks_5_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf327, reinterpret_tensor(primals_171, (384, 384), (1, 384), 0), out=buf329)
        buf330 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf328, buf330, 605184, grid=grid(605184), stream=stream0)
        buf331 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf328, buf331, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf332 = buf277; del buf277  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf331, (48, 64, 197), (12608, 197, 1), 0), out=buf332)
        buf335 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_33, attn_34], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf332, buf335, 9456, 197, grid=grid(9456), stream=stream0)
        buf336 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf329, buf336, 605184, grid=grid(605184), stream=stream0)
        buf337 = reinterpret_tensor(buf329, (48, 197, 64), (12608, 64, 1), 0); del buf329  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf336, (48, 197, 64), (12608, 64, 1), 0), out=buf337)
        buf338 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf337, buf338, 605184, grid=grid(605184), stream=stream0)
        buf339 = reinterpret_tensor(buf337, (1576, 384), (384, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf338, reinterpret_tensor(primals_172, (384, 384), (1, 384), 0), out=buf339)
        buf343 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf344 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf705 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_5_norm_mlp, patch_embed_25, x_107], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf322, buf339, primals_173, primals_174, primals_175, buf343, buf344, buf705, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_175
        buf345 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf344, reinterpret_tensor(primals_176, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf345)
        del primals_177
        buf346 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108, x_111], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf345, buf346, 2420736, grid=grid(2420736), stream=stream0)
        buf347 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf346, reinterpret_tensor(primals_178, (1536, 384), (1, 1536), 0), out=buf347)
        buf348 = reinterpret_tensor(buf347, (8, 197, 384), (75648, 384, 1), 0); del buf347  # reuse
        # Source Nodes: [patch_embed_25, patch_embed_27], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf348, buf322, buf339, primals_173, primals_179, 605184, grid=grid(605184), stream=stream0)
        del primals_173
        del primals_179
        buf350 = buf295; del buf295  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf349, reinterpret_tensor(primals_182, (24, 48), (1, 24), 0), out=buf350)
        buf351 = reinterpret_tensor(buf321, (25088, 24), (24, 1), 0); del buf321  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf349, reinterpret_tensor(primals_183, (24, 24), (1, 24), 0), out=buf351)
        buf352 = reinterpret_tensor(buf260, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf260  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf350, buf352, 602112, grid=grid(602112), stream=stream0)
        buf353 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf350, buf353, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf354 = buf299; del buf299  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf352, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf353, (6272, 6, 16), (96, 16, 1), 0), out=buf354)
        buf357 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_36, attn_37], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf354, buf357, 100352, 16, grid=grid(100352), stream=stream0)
        buf358 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf351, buf358, 602112, grid=grid(602112), stream=stream0)
        buf359 = reinterpret_tensor(buf351, (6272, 16, 6), (96, 6, 1), 0); del buf351  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf357, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf358, (6272, 16, 6), (96, 6, 1), 0), out=buf359)
        buf360 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf359, buf360, 602112, grid=grid(602112), stream=stream0)
        buf361 = reinterpret_tensor(buf359, (25088, 24), (24, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf360, reinterpret_tensor(primals_184, (24, 24), (1, 24), 0), out=buf361)
        buf365 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf366 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf703 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_norm_mlp_in, pixel_embed_19, x_116], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf315, buf361, primals_185, primals_186, primals_187, buf365, buf366, buf703, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_187
        buf367 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf366, reinterpret_tensor(primals_188, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf367)
        del primals_189
        buf368 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117, x_120], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf367, buf368, 2408448, grid=grid(2408448), stream=stream0)
        buf369 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf368, reinterpret_tensor(primals_190, (96, 24), (1, 96), 0), out=buf369)
        buf370 = reinterpret_tensor(buf369, (1568, 16, 24), (384, 24, 1), 0); del buf369  # reuse
        buf374 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf375 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf404 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf701 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_norm1_proj, l__mod___blocks_6_proj, l__mod___blocks_7_attn_in_qk, l__mod___blocks_7_norm_in, pixel_embed_19, pixel_embed_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf370, buf315, buf361, primals_185, primals_191, primals_192, primals_193, primals_208, primals_209, buf374, buf375, buf404, buf701, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_185
        del primals_191
        del primals_193
        del primals_209
        buf376 = reinterpret_tensor(buf361, (1568, 384), (384, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf375, reinterpret_tensor(primals_194, (384, 384), (1, 384), 0), out=buf376)
        buf377 = reinterpret_tensor(buf339, (8, 197, 384), (75648, 384, 1), 0); del buf339  # reuse
        buf378 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf379 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf381 = reinterpret_tensor(buf379, (8, 197, 1), (197, 1, 1), 0); del buf379  # reuse
        buf382 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_18, l__mod___blocks_6_attn_out_qk, l__mod___blocks_6_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf381, buf348, buf376, primals_195, primals_196, primals_197, buf377, buf378, buf382, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_195
        del primals_197
        buf383 = buf328; del buf328  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf382, reinterpret_tensor(primals_198, (384, 768), (1, 384), 0), out=buf383)
        buf384 = reinterpret_tensor(buf348, (1576, 384), (384, 1), 0); del buf348  # reuse
        # Source Nodes: [l__mod___blocks_6_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf382, reinterpret_tensor(primals_199, (384, 384), (1, 384), 0), out=buf384)
        buf385 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf383, buf385, 605184, grid=grid(605184), stream=stream0)
        buf386 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf383, buf386, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf387 = buf332; del buf332  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf385, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf386, (48, 64, 197), (12608, 197, 1), 0), out=buf387)
        buf390 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_39, attn_40], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf387, buf390, 9456, 197, grid=grid(9456), stream=stream0)
        buf391 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf384, buf391, 605184, grid=grid(605184), stream=stream0)
        buf392 = reinterpret_tensor(buf384, (48, 197, 64), (12608, 64, 1), 0); del buf384  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf390, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf391, (48, 197, 64), (12608, 64, 1), 0), out=buf392)
        buf393 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf392, buf393, 605184, grid=grid(605184), stream=stream0)
        buf394 = reinterpret_tensor(buf392, (1576, 384), (384, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf393, reinterpret_tensor(primals_200, (384, 384), (1, 384), 0), out=buf394)
        buf398 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf399 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf702 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_6_norm_mlp, patch_embed_29, x_125], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf377, buf394, primals_201, primals_202, primals_203, buf398, buf399, buf702, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_203
        buf400 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_205, buf399, reinterpret_tensor(primals_204, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf400)
        del primals_205
        buf401 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126, x_129], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf400, buf401, 2420736, grid=grid(2420736), stream=stream0)
        buf402 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf401, reinterpret_tensor(primals_206, (1536, 384), (1, 1536), 0), out=buf402)
        buf403 = reinterpret_tensor(buf402, (8, 197, 384), (75648, 384, 1), 0); del buf402  # reuse
        # Source Nodes: [patch_embed_29, patch_embed_31], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf403, buf377, buf394, primals_201, primals_207, 605184, grid=grid(605184), stream=stream0)
        del primals_201
        del primals_207
        buf405 = buf350; del buf350  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf404, reinterpret_tensor(primals_210, (24, 48), (1, 24), 0), out=buf405)
        buf406 = reinterpret_tensor(buf376, (25088, 24), (24, 1), 0); del buf376  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf404, reinterpret_tensor(primals_211, (24, 24), (1, 24), 0), out=buf406)
        buf407 = reinterpret_tensor(buf315, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf315  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf405, buf407, 602112, grid=grid(602112), stream=stream0)
        buf408 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf405, buf408, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf409 = buf354; del buf354  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf407, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf408, (6272, 6, 16), (96, 16, 1), 0), out=buf409)
        buf412 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_42, attn_43], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf409, buf412, 100352, 16, grid=grid(100352), stream=stream0)
        buf413 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf406, buf413, 602112, grid=grid(602112), stream=stream0)
        buf414 = reinterpret_tensor(buf406, (6272, 16, 6), (96, 6, 1), 0); del buf406  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf412, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf413, (6272, 16, 6), (96, 6, 1), 0), out=buf414)
        buf415 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf414, buf415, 602112, grid=grid(602112), stream=stream0)
        buf416 = reinterpret_tensor(buf414, (25088, 24), (24, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf415, reinterpret_tensor(primals_212, (24, 24), (1, 24), 0), out=buf416)
        buf420 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf421 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf700 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_norm_mlp_in, pixel_embed_22, x_134], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf370, buf416, primals_213, primals_214, primals_215, buf420, buf421, buf700, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_215
        buf422 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_217, buf421, reinterpret_tensor(primals_216, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf422)
        del primals_217
        buf423 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_138], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf422, buf423, 2408448, grid=grid(2408448), stream=stream0)
        buf424 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf423, reinterpret_tensor(primals_218, (96, 24), (1, 96), 0), out=buf424)
        buf425 = reinterpret_tensor(buf424, (1568, 16, 24), (384, 24, 1), 0); del buf424  # reuse
        buf429 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf430 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf459 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf698 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_norm1_proj, l__mod___blocks_7_proj, l__mod___blocks_8_attn_in_qk, l__mod___blocks_8_norm_in, pixel_embed_22, pixel_embed_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf425, buf370, buf416, primals_213, primals_219, primals_220, primals_221, primals_236, primals_237, buf429, buf430, buf459, buf698, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_213
        del primals_219
        del primals_221
        del primals_237
        buf431 = reinterpret_tensor(buf416, (1568, 384), (384, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf430, reinterpret_tensor(primals_222, (384, 384), (1, 384), 0), out=buf431)
        buf432 = reinterpret_tensor(buf394, (8, 197, 384), (75648, 384, 1), 0); del buf394  # reuse
        buf433 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf434 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf436 = reinterpret_tensor(buf434, (8, 197, 1), (197, 1, 1), 0); del buf434  # reuse
        buf437 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_17, l__mod___blocks_7_attn_out_qk, l__mod___blocks_7_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf436, buf403, buf431, primals_223, primals_224, primals_225, buf432, buf433, buf437, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_223
        del primals_225
        buf438 = buf383; del buf383  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf437, reinterpret_tensor(primals_226, (384, 768), (1, 384), 0), out=buf438)
        buf439 = reinterpret_tensor(buf403, (1576, 384), (384, 1), 0); del buf403  # reuse
        # Source Nodes: [l__mod___blocks_7_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf437, reinterpret_tensor(primals_227, (384, 384), (1, 384), 0), out=buf439)
        buf440 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf438, buf440, 605184, grid=grid(605184), stream=stream0)
        buf441 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf438, buf441, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf442 = buf387; del buf387  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf440, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf441, (48, 64, 197), (12608, 197, 1), 0), out=buf442)
        buf445 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_45, attn_46], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf442, buf445, 9456, 197, grid=grid(9456), stream=stream0)
        buf446 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf439, buf446, 605184, grid=grid(605184), stream=stream0)
        buf447 = reinterpret_tensor(buf439, (48, 197, 64), (12608, 64, 1), 0); del buf439  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf446, (48, 197, 64), (12608, 64, 1), 0), out=buf447)
        buf448 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf447, buf448, 605184, grid=grid(605184), stream=stream0)
        buf449 = reinterpret_tensor(buf447, (1576, 384), (384, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf448, reinterpret_tensor(primals_228, (384, 384), (1, 384), 0), out=buf449)
        buf453 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf454 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf699 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_7_norm_mlp, patch_embed_33, x_143], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf432, buf449, primals_229, primals_230, primals_231, buf453, buf454, buf699, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_231
        buf455 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_233, buf454, reinterpret_tensor(primals_232, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf455)
        del primals_233
        buf456 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144, x_147], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf455, buf456, 2420736, grid=grid(2420736), stream=stream0)
        buf457 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf456, reinterpret_tensor(primals_234, (1536, 384), (1, 1536), 0), out=buf457)
        buf458 = reinterpret_tensor(buf457, (8, 197, 384), (75648, 384, 1), 0); del buf457  # reuse
        # Source Nodes: [patch_embed_33, patch_embed_35], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf458, buf432, buf449, primals_229, primals_235, 605184, grid=grid(605184), stream=stream0)
        del primals_229
        del primals_235
        buf460 = buf405; del buf405  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf459, reinterpret_tensor(primals_238, (24, 48), (1, 24), 0), out=buf460)
        buf461 = reinterpret_tensor(buf431, (25088, 24), (24, 1), 0); del buf431  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf459, reinterpret_tensor(primals_239, (24, 24), (1, 24), 0), out=buf461)
        buf462 = reinterpret_tensor(buf370, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf370  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf460, buf462, 602112, grid=grid(602112), stream=stream0)
        buf463 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf460, buf463, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf464 = buf409; del buf409  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf463, (6272, 6, 16), (96, 16, 1), 0), out=buf464)
        buf467 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_48, attn_49], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf464, buf467, 100352, 16, grid=grid(100352), stream=stream0)
        buf468 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf461, buf468, 602112, grid=grid(602112), stream=stream0)
        buf469 = reinterpret_tensor(buf461, (6272, 16, 6), (96, 6, 1), 0); del buf461  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf467, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf468, (6272, 16, 6), (96, 6, 1), 0), out=buf469)
        buf470 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf469, buf470, 602112, grid=grid(602112), stream=stream0)
        buf471 = reinterpret_tensor(buf469, (25088, 24), (24, 1), 0); del buf469  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf470, reinterpret_tensor(primals_240, (24, 24), (1, 24), 0), out=buf471)
        buf475 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf476 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf697 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_norm_mlp_in, pixel_embed_25, x_152], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf425, buf471, primals_241, primals_242, primals_243, buf475, buf476, buf697, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_243
        buf477 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_245, buf476, reinterpret_tensor(primals_244, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf477)
        del primals_245
        buf478 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153, x_156], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf477, buf478, 2408448, grid=grid(2408448), stream=stream0)
        buf479 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf478, reinterpret_tensor(primals_246, (96, 24), (1, 96), 0), out=buf479)
        buf480 = reinterpret_tensor(buf479, (1568, 16, 24), (384, 24, 1), 0); del buf479  # reuse
        buf484 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf485 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf514 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf695 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_norm1_proj, l__mod___blocks_8_proj, l__mod___blocks_9_attn_in_qk, l__mod___blocks_9_norm_in, pixel_embed_25, pixel_embed_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf480, buf425, buf471, primals_241, primals_247, primals_248, primals_249, primals_264, primals_265, buf484, buf485, buf514, buf695, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_241
        del primals_247
        del primals_249
        del primals_265
        buf486 = reinterpret_tensor(buf471, (1568, 384), (384, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf485, reinterpret_tensor(primals_250, (384, 384), (1, 384), 0), out=buf486)
        buf487 = reinterpret_tensor(buf449, (8, 197, 384), (75648, 384, 1), 0); del buf449  # reuse
        buf488 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf489 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf491 = reinterpret_tensor(buf489, (8, 197, 1), (197, 1, 1), 0); del buf489  # reuse
        buf492 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_16, l__mod___blocks_8_attn_out_qk, l__mod___blocks_8_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf491, buf458, buf486, primals_251, primals_252, primals_253, buf487, buf488, buf492, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_251
        del primals_253
        buf493 = buf438; del buf438  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf492, reinterpret_tensor(primals_254, (384, 768), (1, 384), 0), out=buf493)
        buf494 = reinterpret_tensor(buf458, (1576, 384), (384, 1), 0); del buf458  # reuse
        # Source Nodes: [l__mod___blocks_8_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf492, reinterpret_tensor(primals_255, (384, 384), (1, 384), 0), out=buf494)
        buf495 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf493, buf495, 605184, grid=grid(605184), stream=stream0)
        buf496 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf493, buf496, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf497 = buf442; del buf442  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf496, (48, 64, 197), (12608, 197, 1), 0), out=buf497)
        buf500 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_51, attn_52], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf497, buf500, 9456, 197, grid=grid(9456), stream=stream0)
        buf501 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf494, buf501, 605184, grid=grid(605184), stream=stream0)
        buf502 = reinterpret_tensor(buf494, (48, 197, 64), (12608, 64, 1), 0); del buf494  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf500, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf501, (48, 197, 64), (12608, 64, 1), 0), out=buf502)
        buf503 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf502, buf503, 605184, grid=grid(605184), stream=stream0)
        buf504 = reinterpret_tensor(buf502, (1576, 384), (384, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf503, reinterpret_tensor(primals_256, (384, 384), (1, 384), 0), out=buf504)
        buf508 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf509 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf696 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_8_norm_mlp, patch_embed_37, x_161], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf487, buf504, primals_257, primals_258, primals_259, buf508, buf509, buf696, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_259
        buf510 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_261, buf509, reinterpret_tensor(primals_260, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf510)
        del primals_261
        buf511 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162, x_165], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf510, buf511, 2420736, grid=grid(2420736), stream=stream0)
        buf512 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf511, reinterpret_tensor(primals_262, (1536, 384), (1, 1536), 0), out=buf512)
        buf513 = reinterpret_tensor(buf512, (8, 197, 384), (75648, 384, 1), 0); del buf512  # reuse
        # Source Nodes: [patch_embed_37, patch_embed_39], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf513, buf487, buf504, primals_257, primals_263, 605184, grid=grid(605184), stream=stream0)
        del primals_257
        del primals_263
        buf515 = buf460; del buf460  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf514, reinterpret_tensor(primals_266, (24, 48), (1, 24), 0), out=buf515)
        buf516 = reinterpret_tensor(buf486, (25088, 24), (24, 1), 0); del buf486  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf514, reinterpret_tensor(primals_267, (24, 24), (1, 24), 0), out=buf516)
        buf517 = reinterpret_tensor(buf425, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf425  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf515, buf517, 602112, grid=grid(602112), stream=stream0)
        buf518 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf515, buf518, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf519 = buf464; del buf464  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf517, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf518, (6272, 6, 16), (96, 16, 1), 0), out=buf519)
        buf522 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_54, attn_55], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf519, buf522, 100352, 16, grid=grid(100352), stream=stream0)
        buf523 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf516, buf523, 602112, grid=grid(602112), stream=stream0)
        buf524 = reinterpret_tensor(buf516, (6272, 16, 6), (96, 6, 1), 0); del buf516  # reuse
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf522, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf523, (6272, 16, 6), (96, 6, 1), 0), out=buf524)
        buf525 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf524, buf525, 602112, grid=grid(602112), stream=stream0)
        buf526 = reinterpret_tensor(buf524, (25088, 24), (24, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf525, reinterpret_tensor(primals_268, (24, 24), (1, 24), 0), out=buf526)
        buf530 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf531 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf694 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_norm_mlp_in, pixel_embed_28, x_170], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf480, buf526, primals_269, primals_270, primals_271, buf530, buf531, buf694, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_271
        buf532 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_273, buf531, reinterpret_tensor(primals_272, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf532)
        del primals_273
        buf533 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171, x_174], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf532, buf533, 2408448, grid=grid(2408448), stream=stream0)
        buf534 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf533, reinterpret_tensor(primals_274, (96, 24), (1, 96), 0), out=buf534)
        buf535 = reinterpret_tensor(buf534, (1568, 16, 24), (384, 24, 1), 0); del buf534  # reuse
        buf539 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf540 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf569 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf692 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_attn_in_qk, l__mod___blocks_10_norm_in, l__mod___blocks_9_norm1_proj, l__mod___blocks_9_proj, pixel_embed_28, pixel_embed_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf535, buf480, buf526, primals_269, primals_275, primals_276, primals_277, primals_292, primals_293, buf539, buf540, buf569, buf692, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_269
        del primals_275
        del primals_277
        del primals_293
        buf541 = reinterpret_tensor(buf526, (1568, 384), (384, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf540, reinterpret_tensor(primals_278, (384, 384), (1, 384), 0), out=buf541)
        buf542 = reinterpret_tensor(buf504, (8, 197, 384), (75648, 384, 1), 0); del buf504  # reuse
        buf543 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf544 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf546 = reinterpret_tensor(buf544, (8, 197, 1), (197, 1, 1), 0); del buf544  # reuse
        buf547 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_15, l__mod___blocks_9_attn_out_qk, l__mod___blocks_9_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf546, buf513, buf541, primals_279, primals_280, primals_281, buf542, buf543, buf547, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_279
        del primals_281
        buf548 = buf493; del buf493  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf547, reinterpret_tensor(primals_282, (384, 768), (1, 384), 0), out=buf548)
        buf549 = reinterpret_tensor(buf513, (1576, 384), (384, 1), 0); del buf513  # reuse
        # Source Nodes: [l__mod___blocks_9_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf547, reinterpret_tensor(primals_283, (384, 384), (1, 384), 0), out=buf549)
        buf550 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf548, buf550, 605184, grid=grid(605184), stream=stream0)
        buf551 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf548, buf551, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf552 = buf497; del buf497  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf550, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf551, (48, 64, 197), (12608, 197, 1), 0), out=buf552)
        buf555 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_57, attn_58], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf552, buf555, 9456, 197, grid=grid(9456), stream=stream0)
        buf556 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf549, buf556, 605184, grid=grid(605184), stream=stream0)
        buf557 = reinterpret_tensor(buf549, (48, 197, 64), (12608, 64, 1), 0); del buf549  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf555, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf556, (48, 197, 64), (12608, 64, 1), 0), out=buf557)
        buf558 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf557, buf558, 605184, grid=grid(605184), stream=stream0)
        buf559 = reinterpret_tensor(buf557, (1576, 384), (384, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf558, reinterpret_tensor(primals_284, (384, 384), (1, 384), 0), out=buf559)
        buf563 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf564 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf693 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_9_norm_mlp, patch_embed_41, x_179], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf542, buf559, primals_285, primals_286, primals_287, buf563, buf564, buf693, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_287
        buf565 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_289, buf564, reinterpret_tensor(primals_288, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf565)
        del primals_289
        buf566 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_180, x_183], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf565, buf566, 2420736, grid=grid(2420736), stream=stream0)
        buf567 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf566, reinterpret_tensor(primals_290, (1536, 384), (1, 1536), 0), out=buf567)
        buf568 = reinterpret_tensor(buf567, (8, 197, 384), (75648, 384, 1), 0); del buf567  # reuse
        # Source Nodes: [patch_embed_41, patch_embed_43], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf568, buf542, buf559, primals_285, primals_291, 605184, grid=grid(605184), stream=stream0)
        del primals_285
        del primals_291
        buf570 = buf515; del buf515  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf569, reinterpret_tensor(primals_294, (24, 48), (1, 24), 0), out=buf570)
        buf571 = reinterpret_tensor(buf541, (25088, 24), (24, 1), 0); del buf541  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf569, reinterpret_tensor(primals_295, (24, 24), (1, 24), 0), out=buf571)
        buf572 = reinterpret_tensor(buf480, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf480  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf570, buf572, 602112, grid=grid(602112), stream=stream0)
        buf573 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf570, buf573, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf574 = buf519; del buf519  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf572, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf573, (6272, 6, 16), (96, 16, 1), 0), out=buf574)
        buf577 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_60, attn_61], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf574, buf577, 100352, 16, grid=grid(100352), stream=stream0)
        buf578 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf571, buf578, 602112, grid=grid(602112), stream=stream0)
        buf579 = reinterpret_tensor(buf571, (6272, 16, 6), (96, 6, 1), 0); del buf571  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf578, (6272, 16, 6), (96, 6, 1), 0), out=buf579)
        buf580 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf579, buf580, 602112, grid=grid(602112), stream=stream0)
        buf581 = reinterpret_tensor(buf579, (25088, 24), (24, 1), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf580, reinterpret_tensor(primals_296, (24, 24), (1, 24), 0), out=buf581)
        buf585 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf586 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf691 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_norm_mlp_in, pixel_embed_31, x_188], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf535, buf581, primals_297, primals_298, primals_299, buf585, buf586, buf691, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_299
        buf587 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_301, buf586, reinterpret_tensor(primals_300, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf587)
        del primals_301
        buf588 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189, x_192], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf587, buf588, 2408448, grid=grid(2408448), stream=stream0)
        buf589 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf588, reinterpret_tensor(primals_302, (96, 24), (1, 96), 0), out=buf589)
        buf590 = reinterpret_tensor(buf589, (1568, 16, 24), (384, 24, 1), 0); del buf589  # reuse
        buf594 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf595 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf624 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf689 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_norm1_proj, l__mod___blocks_10_proj, l__mod___blocks_11_attn_in_qk, l__mod___blocks_11_norm_in, pixel_embed_31, pixel_embed_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_24.run(buf590, buf535, buf581, primals_297, primals_303, primals_304, primals_305, primals_320, primals_321, buf594, buf595, buf624, buf689, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_297
        del primals_303
        del primals_305
        del primals_321
        buf596 = reinterpret_tensor(buf581, (1568, 384), (384, 1), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf595, reinterpret_tensor(primals_306, (384, 384), (1, 384), 0), out=buf596)
        buf597 = reinterpret_tensor(buf559, (8, 197, 384), (75648, 384, 1), 0); del buf559  # reuse
        buf598 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf599 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf601 = reinterpret_tensor(buf599, (8, 197, 1), (197, 1, 1), 0); del buf599  # reuse
        buf602 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_14, l__mod___blocks_10_attn_out_qk, l__mod___blocks_10_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf601, buf568, buf596, primals_307, primals_308, primals_309, buf597, buf598, buf602, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_307
        del primals_309
        buf603 = buf548; del buf548  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf602, reinterpret_tensor(primals_310, (384, 768), (1, 384), 0), out=buf603)
        buf604 = reinterpret_tensor(buf568, (1576, 384), (384, 1), 0); del buf568  # reuse
        # Source Nodes: [l__mod___blocks_10_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf602, reinterpret_tensor(primals_311, (384, 384), (1, 384), 0), out=buf604)
        buf605 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf603, buf605, 605184, grid=grid(605184), stream=stream0)
        buf606 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf603, buf606, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf607 = buf552; del buf552  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf605, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf606, (48, 64, 197), (12608, 197, 1), 0), out=buf607)
        buf610 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_63, attn_64], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf607, buf610, 9456, 197, grid=grid(9456), stream=stream0)
        buf611 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf604, buf611, 605184, grid=grid(605184), stream=stream0)
        buf612 = reinterpret_tensor(buf604, (48, 197, 64), (12608, 64, 1), 0); del buf604  # reuse
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf610, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf611, (48, 197, 64), (12608, 64, 1), 0), out=buf612)
        buf613 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf612, buf613, 605184, grid=grid(605184), stream=stream0)
        buf614 = reinterpret_tensor(buf612, (1576, 384), (384, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf613, reinterpret_tensor(primals_312, (384, 384), (1, 384), 0), out=buf614)
        buf618 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf619 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf690 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_10_norm_mlp, patch_embed_45, x_197], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf597, buf614, primals_313, primals_314, primals_315, buf618, buf619, buf690, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_315
        buf620 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_317, buf619, reinterpret_tensor(primals_316, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf620)
        del primals_317
        buf621 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_198, x_201], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf620, buf621, 2420736, grid=grid(2420736), stream=stream0)
        buf622 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf621, reinterpret_tensor(primals_318, (1536, 384), (1, 1536), 0), out=buf622)
        buf623 = reinterpret_tensor(buf622, (8, 197, 384), (75648, 384, 1), 0); del buf622  # reuse
        # Source Nodes: [patch_embed_45, patch_embed_47], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf623, buf597, buf614, primals_313, primals_319, 605184, grid=grid(605184), stream=stream0)
        del primals_313
        del primals_319
        buf625 = buf570; del buf570  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_in_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf624, reinterpret_tensor(primals_322, (24, 48), (1, 24), 0), out=buf625)
        buf626 = reinterpret_tensor(buf596, (25088, 24), (24, 1), 0); del buf596  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_in_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf624, reinterpret_tensor(primals_323, (24, 24), (1, 24), 0), out=buf626)
        buf627 = reinterpret_tensor(buf535, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf535  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf625, buf627, 602112, grid=grid(602112), stream=stream0)
        buf628 = empty((1568, 4, 6, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf625, buf628, 37632, 16, grid=grid(37632, 16), stream=stream0)
        del buf625
        buf629 = buf574; del buf574  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf627, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf628, (6272, 6, 16), (96, 16, 1), 0), out=buf629)
        buf632 = empty((1568, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_66, attn_67], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_8.run(buf629, buf632, 100352, 16, grid=grid(100352), stream=stream0)
        del buf629
        buf633 = empty((1568, 4, 16, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf626, buf633, 602112, grid=grid(602112), stream=stream0)
        buf634 = reinterpret_tensor(buf626, (6272, 16, 6), (96, 6, 1), 0); del buf626  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf632, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf633, (6272, 16, 6), (96, 6, 1), 0), out=buf634)
        buf635 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf634, buf635, 602112, grid=grid(602112), stream=stream0)
        buf636 = reinterpret_tensor(buf634, (25088, 24), (24, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf635, reinterpret_tensor(primals_324, (24, 24), (1, 24), 0), out=buf636)
        buf640 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf641 = empty((25088, 24), device='cuda', dtype=torch.float32)
        buf688 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_norm_mlp_in, pixel_embed_34, x_206], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_23.run(buf590, buf636, primals_325, primals_326, primals_327, buf640, buf641, buf688, 25088, 24, grid=grid(25088), stream=stream0)
        del primals_327
        buf642 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_329, buf641, reinterpret_tensor(primals_328, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf642)
        del primals_329
        buf643 = empty((25088, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207, x_210], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf642, buf643, 2408448, grid=grid(2408448), stream=stream0)
        buf644 = empty((25088, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf643, reinterpret_tensor(primals_330, (96, 24), (1, 96), 0), out=buf644)
        buf645 = reinterpret_tensor(buf644, (1568, 16, 24), (384, 24, 1), 0); del buf644  # reuse
        buf649 = empty((1568, 16, 24), device='cuda', dtype=torch.float32)
        buf650 = empty((1568, 384), device='cuda', dtype=torch.float32)
        buf687 = empty((1568, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_norm1_proj, l__mod___blocks_11_proj, pixel_embed_34, pixel_embed_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_25.run(buf645, buf590, buf636, primals_325, primals_331, primals_332, primals_333, buf649, buf650, buf687, 25088, 24, grid=grid(25088), stream=stream0)
        del buf590
        del buf636
        del primals_325
        del primals_331
        del primals_333
        buf651 = reinterpret_tensor(buf645, (1568, 384), (384, 1), 0); del buf645  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf650, reinterpret_tensor(primals_334, (384, 384), (1, 384), 0), out=buf651)
        buf652 = reinterpret_tensor(buf614, (8, 197, 384), (75648, 384, 1), 0); del buf614  # reuse
        buf653 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf654 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf656 = reinterpret_tensor(buf654, (8, 197, 1), (197, 1, 1), 0); del buf654  # reuse
        buf657 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_13, l__mod___blocks_11_attn_out_qk, l__mod___blocks_11_norm_out], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_14.run(buf656, buf623, buf651, primals_335, primals_336, primals_337, buf652, buf653, buf657, 1576, 384, grid=grid(1576), stream=stream0)
        del buf651
        del primals_335
        del primals_337
        buf658 = buf603; del buf603  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_out_qk], Original ATen: [aten.mm]
        extern_kernels.mm(buf657, reinterpret_tensor(primals_338, (384, 768), (1, 384), 0), out=buf658)
        buf659 = reinterpret_tensor(buf623, (1576, 384), (384, 1), 0); del buf623  # reuse
        # Source Nodes: [l__mod___blocks_11_attn_out_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf657, reinterpret_tensor(primals_339, (384, 384), (1, 384), 0), out=buf659)
        buf660 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf658, buf660, 605184, grid=grid(605184), stream=stream0)
        buf661 = empty((8, 6, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf658, buf661, 3072, 197, grid=grid(3072, 197), stream=stream0)
        del buf658
        buf662 = buf607; del buf607  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf660, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf661, (48, 64, 197), (12608, 197, 1), 0), out=buf662)
        buf665 = empty((8, 6, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_69, attn_70], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_17.run(buf662, buf665, 9456, 197, grid=grid(9456), stream=stream0)
        del buf662
        buf666 = empty((8, 6, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf659, buf666, 605184, grid=grid(605184), stream=stream0)
        buf667 = reinterpret_tensor(buf659, (48, 197, 64), (12608, 64, 1), 0); del buf659  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf665, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf666, (48, 197, 64), (12608, 64, 1), 0), out=buf667)
        buf668 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf667, buf668, 605184, grid=grid(605184), stream=stream0)
        buf669 = reinterpret_tensor(buf667, (1576, 384), (384, 1), 0); del buf667  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf668, reinterpret_tensor(primals_340, (384, 384), (1, 384), 0), out=buf669)
        buf673 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf674 = empty((1576, 384), device='cuda', dtype=torch.float32)
        buf686 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_11_norm_mlp, patch_embed_49, x_215], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_20.run(buf652, buf669, primals_341, primals_342, primals_343, buf673, buf674, buf686, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_343
        buf675 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_345, buf674, reinterpret_tensor(primals_344, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf675)
        del primals_345
        buf676 = empty((1576, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216, x_219], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_21.run(buf675, buf676, 2420736, grid=grid(2420736), stream=stream0)
        buf677 = empty((1576, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf676, reinterpret_tensor(primals_346, (1536, 384), (1, 1536), 0), out=buf677)
        buf678 = reinterpret_tensor(buf677, (8, 197, 384), (75648, 384, 1), 0); del buf677  # reuse
        buf682 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        buf685 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [patch_embed_49, patch_embed_51, x_221], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_26.run(buf678, buf652, buf669, primals_341, primals_347, buf682, buf685, 1576, 384, grid=grid(1576), stream=stream0)
        del buf669
        del buf678
        del primals_341
        del primals_347
        buf683 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf682, primals_348, primals_349, buf683, 3072, grid=grid(3072), stream=stream0)
        del primals_349
        buf684 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_351, buf683, reinterpret_tensor(primals_350, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf684)
        del primals_351
        return (buf684, primals_4, primals_6, primals_10, primals_12, primals_18, primals_24, primals_28, primals_34, primals_40, primals_46, primals_52, primals_56, primals_62, primals_68, primals_74, primals_80, primals_84, primals_90, primals_96, primals_102, primals_108, primals_112, primals_118, primals_124, primals_130, primals_136, primals_140, primals_146, primals_152, primals_158, primals_164, primals_168, primals_174, primals_180, primals_186, primals_192, primals_196, primals_202, primals_208, primals_214, primals_220, primals_224, primals_230, primals_236, primals_242, primals_248, primals_252, primals_258, primals_264, primals_270, primals_276, primals_280, primals_286, primals_292, primals_298, primals_304, primals_308, primals_314, primals_320, primals_326, primals_332, primals_336, primals_342, primals_348, primals_352, buf1, reinterpret_tensor(buf1, (4, 14, 1, 1), (14, 1, 1, 1), 0), buf3, buf4, buf7, buf8, buf9, buf10, buf13, buf15, buf18, buf19, buf30, buf35, buf36, buf37, buf38, buf44, buf45, buf47, buf48, buf51, buf52, buf63, buf68, buf69, buf70, buf71, buf74, buf85, buf90, buf91, buf92, buf93, buf99, buf100, buf102, buf103, buf106, buf107, buf118, buf123, buf124, buf125, buf126, buf129, buf140, buf145, buf146, buf147, buf148, buf154, buf155, buf157, buf158, buf161, buf162, buf173, buf178, buf179, buf180, buf181, buf184, buf195, buf200, buf201, buf202, buf203, buf209, buf210, buf212, buf213, buf216, buf217, buf228, buf233, buf234, buf235, buf236, buf239, buf250, buf255, buf256, buf257, buf258, buf264, buf265, buf267, buf268, buf271, buf272, buf283, buf288, buf289, buf290, buf291, buf294, buf305, buf310, buf311, buf312, buf313, buf319, buf320, buf322, buf323, buf326, buf327, buf338, buf343, buf344, buf345, buf346, buf349, buf360, buf365, buf366, buf367, buf368, buf374, buf375, buf377, buf378, buf381, buf382, buf393, buf398, buf399, buf400, buf401, buf404, buf415, buf420, buf421, buf422, buf423, buf429, buf430, buf432, buf433, buf436, buf437, buf448, buf453, buf454, buf455, buf456, buf459, buf470, buf475, buf476, buf477, buf478, buf484, buf485, buf487, buf488, buf491, buf492, buf503, buf508, buf509, buf510, buf511, buf514, buf525, buf530, buf531, buf532, buf533, buf539, buf540, buf542, buf543, buf546, buf547, buf558, buf563, buf564, buf565, buf566, buf569, buf580, buf585, buf586, buf587, buf588, buf594, buf595, buf597, buf598, buf601, buf602, buf613, buf618, buf619, buf620, buf621, buf624, buf635, buf640, buf641, buf642, buf643, buf649, buf650, buf652, buf653, buf656, buf657, buf668, buf673, buf674, buf675, buf676, buf682, buf683, reinterpret_tensor(primals_350, (1000, 384), (384, 1), 0), buf685, reinterpret_tensor(primals_346, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_344, (1536, 384), (384, 1), 0), buf686, reinterpret_tensor(primals_340, (384, 384), (384, 1), 0), reinterpret_tensor(buf665, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf666, (48, 64, 197), (12608, 1, 64), 0), buf665, reinterpret_tensor(buf660, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf661, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_339, (384, 384), (384, 1), 0), reinterpret_tensor(primals_338, (768, 384), (384, 1), 0), reinterpret_tensor(primals_334, (384, 384), (384, 1), 0), buf687, reinterpret_tensor(primals_330, (24, 96), (96, 1), 0), reinterpret_tensor(primals_328, (96, 24), (24, 1), 0), buf688, reinterpret_tensor(primals_324, (24, 24), (24, 1), 0), reinterpret_tensor(buf632, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf633, (6272, 6, 16), (96, 1, 6), 0), buf632, reinterpret_tensor(buf627, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf628, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_323, (24, 24), (24, 1), 0), reinterpret_tensor(primals_322, (48, 24), (24, 1), 0), buf689, reinterpret_tensor(primals_318, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_316, (1536, 384), (384, 1), 0), buf690, reinterpret_tensor(primals_312, (384, 384), (384, 1), 0), reinterpret_tensor(buf610, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf611, (48, 64, 197), (12608, 1, 64), 0), buf610, reinterpret_tensor(buf605, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf606, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_311, (384, 384), (384, 1), 0), reinterpret_tensor(primals_310, (768, 384), (384, 1), 0), reinterpret_tensor(primals_306, (384, 384), (384, 1), 0), reinterpret_tensor(primals_302, (24, 96), (96, 1), 0), reinterpret_tensor(primals_300, (96, 24), (24, 1), 0), buf691, reinterpret_tensor(primals_296, (24, 24), (24, 1), 0), reinterpret_tensor(buf577, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf578, (6272, 6, 16), (96, 1, 6), 0), buf577, reinterpret_tensor(buf572, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf573, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_295, (24, 24), (24, 1), 0), reinterpret_tensor(primals_294, (48, 24), (24, 1), 0), buf692, reinterpret_tensor(primals_290, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_288, (1536, 384), (384, 1), 0), buf693, reinterpret_tensor(primals_284, (384, 384), (384, 1), 0), reinterpret_tensor(buf555, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf556, (48, 64, 197), (12608, 1, 64), 0), buf555, reinterpret_tensor(buf550, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf551, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_283, (384, 384), (384, 1), 0), reinterpret_tensor(primals_282, (768, 384), (384, 1), 0), reinterpret_tensor(primals_278, (384, 384), (384, 1), 0), reinterpret_tensor(primals_274, (24, 96), (96, 1), 0), reinterpret_tensor(primals_272, (96, 24), (24, 1), 0), buf694, reinterpret_tensor(primals_268, (24, 24), (24, 1), 0), reinterpret_tensor(buf522, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf523, (6272, 6, 16), (96, 1, 6), 0), buf522, reinterpret_tensor(buf517, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf518, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_267, (24, 24), (24, 1), 0), reinterpret_tensor(primals_266, (48, 24), (24, 1), 0), buf695, reinterpret_tensor(primals_262, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_260, (1536, 384), (384, 1), 0), buf696, reinterpret_tensor(primals_256, (384, 384), (384, 1), 0), reinterpret_tensor(buf500, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf501, (48, 64, 197), (12608, 1, 64), 0), buf500, reinterpret_tensor(buf495, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf496, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_255, (384, 384), (384, 1), 0), reinterpret_tensor(primals_254, (768, 384), (384, 1), 0), reinterpret_tensor(primals_250, (384, 384), (384, 1), 0), reinterpret_tensor(primals_246, (24, 96), (96, 1), 0), reinterpret_tensor(primals_244, (96, 24), (24, 1), 0), buf697, reinterpret_tensor(primals_240, (24, 24), (24, 1), 0), reinterpret_tensor(buf467, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf468, (6272, 6, 16), (96, 1, 6), 0), buf467, reinterpret_tensor(buf462, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf463, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_239, (24, 24), (24, 1), 0), reinterpret_tensor(primals_238, (48, 24), (24, 1), 0), buf698, reinterpret_tensor(primals_234, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_232, (1536, 384), (384, 1), 0), buf699, reinterpret_tensor(primals_228, (384, 384), (384, 1), 0), reinterpret_tensor(buf445, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf446, (48, 64, 197), (12608, 1, 64), 0), buf445, reinterpret_tensor(buf440, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf441, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_227, (384, 384), (384, 1), 0), reinterpret_tensor(primals_226, (768, 384), (384, 1), 0), reinterpret_tensor(primals_222, (384, 384), (384, 1), 0), reinterpret_tensor(primals_218, (24, 96), (96, 1), 0), reinterpret_tensor(primals_216, (96, 24), (24, 1), 0), buf700, reinterpret_tensor(primals_212, (24, 24), (24, 1), 0), reinterpret_tensor(buf412, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf413, (6272, 6, 16), (96, 1, 6), 0), buf412, reinterpret_tensor(buf407, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf408, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_211, (24, 24), (24, 1), 0), reinterpret_tensor(primals_210, (48, 24), (24, 1), 0), buf701, reinterpret_tensor(primals_206, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_204, (1536, 384), (384, 1), 0), buf702, reinterpret_tensor(primals_200, (384, 384), (384, 1), 0), reinterpret_tensor(buf390, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf391, (48, 64, 197), (12608, 1, 64), 0), buf390, reinterpret_tensor(buf385, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf386, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_199, (384, 384), (384, 1), 0), reinterpret_tensor(primals_198, (768, 384), (384, 1), 0), reinterpret_tensor(primals_194, (384, 384), (384, 1), 0), reinterpret_tensor(primals_190, (24, 96), (96, 1), 0), reinterpret_tensor(primals_188, (96, 24), (24, 1), 0), buf703, reinterpret_tensor(primals_184, (24, 24), (24, 1), 0), reinterpret_tensor(buf357, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf358, (6272, 6, 16), (96, 1, 6), 0), buf357, reinterpret_tensor(buf352, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf353, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_183, (24, 24), (24, 1), 0), reinterpret_tensor(primals_182, (48, 24), (24, 1), 0), buf704, reinterpret_tensor(primals_178, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_176, (1536, 384), (384, 1), 0), buf705, reinterpret_tensor(primals_172, (384, 384), (384, 1), 0), reinterpret_tensor(buf335, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf336, (48, 64, 197), (12608, 1, 64), 0), buf335, reinterpret_tensor(buf330, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf331, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_171, (384, 384), (384, 1), 0), reinterpret_tensor(primals_170, (768, 384), (384, 1), 0), reinterpret_tensor(primals_166, (384, 384), (384, 1), 0), reinterpret_tensor(primals_162, (24, 96), (96, 1), 0), reinterpret_tensor(primals_160, (96, 24), (24, 1), 0), buf706, reinterpret_tensor(primals_156, (24, 24), (24, 1), 0), reinterpret_tensor(buf302, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf303, (6272, 6, 16), (96, 1, 6), 0), buf302, reinterpret_tensor(buf297, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf298, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_155, (24, 24), (24, 1), 0), reinterpret_tensor(primals_154, (48, 24), (24, 1), 0), buf707, reinterpret_tensor(primals_150, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_148, (1536, 384), (384, 1), 0), buf708, reinterpret_tensor(primals_144, (384, 384), (384, 1), 0), reinterpret_tensor(buf280, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf281, (48, 64, 197), (12608, 1, 64), 0), buf280, reinterpret_tensor(buf275, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf276, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_143, (384, 384), (384, 1), 0), reinterpret_tensor(primals_142, (768, 384), (384, 1), 0), reinterpret_tensor(primals_138, (384, 384), (384, 1), 0), reinterpret_tensor(primals_134, (24, 96), (96, 1), 0), reinterpret_tensor(primals_132, (96, 24), (24, 1), 0), buf709, reinterpret_tensor(primals_128, (24, 24), (24, 1), 0), reinterpret_tensor(buf247, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf248, (6272, 6, 16), (96, 1, 6), 0), buf247, reinterpret_tensor(buf242, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf243, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_127, (24, 24), (24, 1), 0), reinterpret_tensor(primals_126, (48, 24), (24, 1), 0), buf710, reinterpret_tensor(primals_122, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_120, (1536, 384), (384, 1), 0), buf711, reinterpret_tensor(primals_116, (384, 384), (384, 1), 0), reinterpret_tensor(buf225, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf226, (48, 64, 197), (12608, 1, 64), 0), buf225, reinterpret_tensor(buf220, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf221, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_115, (384, 384), (384, 1), 0), reinterpret_tensor(primals_114, (768, 384), (384, 1), 0), reinterpret_tensor(primals_110, (384, 384), (384, 1), 0), reinterpret_tensor(primals_106, (24, 96), (96, 1), 0), reinterpret_tensor(primals_104, (96, 24), (24, 1), 0), buf712, reinterpret_tensor(primals_100, (24, 24), (24, 1), 0), reinterpret_tensor(buf192, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf193, (6272, 6, 16), (96, 1, 6), 0), buf192, reinterpret_tensor(buf187, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf188, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_99, (24, 24), (24, 1), 0), reinterpret_tensor(primals_98, (48, 24), (24, 1), 0), buf713, reinterpret_tensor(primals_94, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_92, (1536, 384), (384, 1), 0), buf714, reinterpret_tensor(primals_88, (384, 384), (384, 1), 0), reinterpret_tensor(buf170, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf171, (48, 64, 197), (12608, 1, 64), 0), buf170, reinterpret_tensor(buf165, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf166, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_87, (384, 384), (384, 1), 0), reinterpret_tensor(primals_86, (768, 384), (384, 1), 0), reinterpret_tensor(primals_82, (384, 384), (384, 1), 0), reinterpret_tensor(primals_78, (24, 96), (96, 1), 0), reinterpret_tensor(primals_76, (96, 24), (24, 1), 0), buf715, reinterpret_tensor(primals_72, (24, 24), (24, 1), 0), reinterpret_tensor(buf137, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf138, (6272, 6, 16), (96, 1, 6), 0), buf137, reinterpret_tensor(buf132, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf133, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_71, (24, 24), (24, 1), 0), reinterpret_tensor(primals_70, (48, 24), (24, 1), 0), buf716, reinterpret_tensor(primals_66, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_64, (1536, 384), (384, 1), 0), buf717, reinterpret_tensor(primals_60, (384, 384), (384, 1), 0), reinterpret_tensor(buf115, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf116, (48, 64, 197), (12608, 1, 64), 0), buf115, reinterpret_tensor(buf110, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf111, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_59, (384, 384), (384, 1), 0), reinterpret_tensor(primals_58, (768, 384), (384, 1), 0), reinterpret_tensor(primals_54, (384, 384), (384, 1), 0), reinterpret_tensor(primals_50, (24, 96), (96, 1), 0), reinterpret_tensor(primals_48, (96, 24), (24, 1), 0), buf718, reinterpret_tensor(primals_44, (24, 24), (24, 1), 0), reinterpret_tensor(buf82, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf83, (6272, 6, 16), (96, 1, 6), 0), buf82, reinterpret_tensor(buf77, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf78, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_43, (24, 24), (24, 1), 0), reinterpret_tensor(primals_42, (48, 24), (24, 1), 0), buf719, reinterpret_tensor(primals_38, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_36, (1536, 384), (384, 1), 0), buf720, reinterpret_tensor(primals_32, (384, 384), (384, 1), 0), reinterpret_tensor(buf60, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf61, (48, 64, 197), (12608, 1, 64), 0), buf60, reinterpret_tensor(buf55, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf56, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_31, (384, 384), (384, 1), 0), reinterpret_tensor(primals_30, (768, 384), (384, 1), 0), reinterpret_tensor(primals_26, (384, 384), (384, 1), 0), reinterpret_tensor(primals_22, (24, 96), (96, 1), 0), reinterpret_tensor(primals_20, (96, 24), (24, 1), 0), buf721, reinterpret_tensor(primals_16, (24, 24), (24, 1), 0), reinterpret_tensor(buf27, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf28, (6272, 6, 16), (96, 1, 6), 0), buf27, reinterpret_tensor(buf22, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf23, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_15, (24, 24), (24, 1), 0), reinterpret_tensor(primals_14, (48, 24), (24, 1), 0), reinterpret_tensor(primals_8, (384, 384), (384, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 24, 4, 4), (384, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tnt_s_patch16_224', benchmark_compiled_module)
