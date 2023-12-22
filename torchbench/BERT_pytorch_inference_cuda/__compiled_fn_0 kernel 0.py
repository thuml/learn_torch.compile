
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


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5isykqjmxeypbj3so5527mip3jqnxmmtbuckggwcavo7wyajht.py
# Source Nodes: [mask], Original ATen: [aten.unsqueeze]
# mask => unsqueeze_1
triton_poi_fused_unsqueeze_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unsqueeze_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x2 = (xindex // 16384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/le/cleoobtu4s4cnsknujxqjs24rphnylrrjpu266q4fjrar2l7b4he.py
# Source Nodes: [add, add_2, forward, forward_1, mean, mul, std, sub, truediv, x, x_4], Original ATen: [aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.std, aten.sub]
# add => add
# add_2 => add_2
# forward => embedding
# forward_1 => embedding_1
# mean => mean
# mul => mul
# std => sqrt, var
# sub => sub
# truediv => div
# x => add_1
# x_4 => add_3
triton_per_fused_add_div_embedding_mean_mul_std_sub_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_mean_mul_std_sub_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 20005
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 20005)) | ~xmask, "index out of bounds: 0 <= tmp3 < 20005")
    tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 + 3
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 3)) | ~xmask, "index out of bounds: 0 <= tmp10 < 3")
    tmp11 = tl.load(in_ptr4 + (r2 + (768*tmp10)), rmask & xmask, other=0.0)
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 768, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp13 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = 768.0
    tmp33 = tmp16 / tmp32
    tmp34 = tmp12 - tmp33
    tmp35 = tmp31 * tmp34
    tmp36 = 767.0
    tmp37 = tmp30 / tmp36
    tmp38 = tl.sqrt(tmp37)
    tmp39 = 1e-06
    tmp40 = tmp38 + tmp39
    tmp41 = tmp35 / tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp12, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp43, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqm2ev6chliwzxdxe3zsjlwh7zi3bvo2zfettbexlfxq2magozl.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_1
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 12
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (98304*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmk4mysfnngwhswc3hlpmo26enmnoah33t4s4e7c65snzfk2zddg.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_2
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4chdrgisrykxzhgg7lehemuaobxlyxnpl55olqnmph7t6gjuxq.py
# Source Nodes: [eq, p_attn, scores, scores_1], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
# eq => eq
# p_attn => amax, div_2, exp, sub_1, sum_1
# scores => div_1
# scores_1 => full_default, where
triton_per_fused__softmax_div_eq_masked_fill_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_eq_masked_fill_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 128
    x2 = (xindex // 1536)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, other=0.0)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp5 = 8.0
    tmp6 = tmp4 / tmp5
    tmp7 = -1000000000.0
    tmp8 = tl.where(tmp3, tmp7, tmp6)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (128*x4)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqquoc5e7tuijpa5jftv3f2izdsh6ktcny3o5xchfrh3szxjb4k.py
# Source Nodes: [contiguous], Original ATen: [aten.clone]
# contiguous => clone_5
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 128
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1) + (98304*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyh364e3hvcrydab23lh5kex3yy3geeim5l6bnrdhj3oo7tk5wi3.py
# Source Nodes: [add_5, add_6, mean_2, mul_1, std_2, sub_1, truediv_2, x_7], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_5 => add_5
# add_6 => add_6
# mean_2 => mean_1
# mul_1 => mul_1
# std_2 => sqrt_1, var_1
# sub_1 => sub_2
# truediv_2 => div_3
# x_7 => add_4
triton_per_fused_add_div_mean_mul_std_sub_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_std_sub_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp23 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp5 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = 768.0
    tmp25 = tmp8 / tmp24
    tmp26 = tmp4 - tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = 767.0
    tmp29 = tmp22 / tmp28
    tmp30 = tl.sqrt(tmp29)
    tmp31 = 1e-06
    tmp32 = tmp30 + tmp31
    tmp33 = tmp27 / tmp32
    tmp35 = tmp33 + tmp34
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpefqbhj5zyhyqgqenvbdpbl47panowqjqgiqw3cljtkj736dvk.py
# Source Nodes: [l__mod___transformer_blocks_0_feed_forward_activation], Original ATen: [aten.gelu]
# l__mod___transformer_blocks_0_feed_forward_activation => add_7, erf, mul_2, mul_3, mul_4
triton_poi_fused_gelu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsshebratpbzpabhmjhjatbzxj24pndmrinikinxpgwga657gb5.py
# Source Nodes: [add_8, mean_4, mul_2, std_4, sub_2, truediv_3, x_12, x_7, x_8], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_8 => add_9
# mean_4 => mean_2
# mul_2 => mul_5
# std_4 => sqrt_2, var_2
# sub_2 => sub_3
# truediv_3 => div_4
# x_12 => add_10
# x_7 => add_4
# x_8 => add_8
triton_per_fused_add_div_mean_mul_std_sub_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_std_sub_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 512
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
    tmp27 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp9 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = 768.0
    tmp29 = tmp12 / tmp28
    tmp30 = tmp8 - tmp29
    tmp31 = tmp27 * tmp30
    tmp32 = 767.0
    tmp33 = tmp26 / tmp32
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1e-06
    tmp36 = tmp34 + tmp35
    tmp37 = tmp31 / tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6qwp6eclr5tvc534lmzjr4pkmt2nj4io3rmsznslz6ti3llrxe.py
# Source Nodes: [x_95, x_96], Original ATen: [aten.add]
# x_95 => add_81
# x_96 => add_85
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, ), (1, ))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (20005, 768), (768, 1))
    assert_size_stride(arg49_1, (3, 768), (768, 1))
    assert_size_stride(arg50_1, (768, 768), (768, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 768), (768, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (3072, 768), (768, 1))
    assert_size_stride(arg59_1, (3072, ), (1, ))
    assert_size_stride(arg60_1, (768, 3072), (3072, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, 768), (768, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, 768), (768, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, 768), (768, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (3072, 768), (768, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 768), (768, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, 768), (768, 1))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, 768), (768, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (3072, 768), (768, 1))
    assert_size_stride(arg83_1, (3072, ), (1, ))
    assert_size_stride(arg84_1, (768, 3072), (3072, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, 768), (768, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (3072, 768), (768, 1))
    assert_size_stride(arg107_1, (3072, ), (1, ))
    assert_size_stride(arg108_1, (768, 3072), (3072, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 768), (768, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, 768), (768, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (3072, 768), (768, 1))
    assert_size_stride(arg119_1, (3072, ), (1, ))
    assert_size_stride(arg120_1, (768, 3072), (3072, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, 768), (768, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, 768), (768, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, 768), (768, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (3072, 768), (768, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (768, 3072), (3072, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, 768), (768, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, 768), (768, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, 768), (768, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768), (768, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (3072, 768), (768, 1))
    assert_size_stride(arg155_1, (3072, ), (1, ))
    assert_size_stride(arg156_1, (768, 3072), (3072, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, 768), (768, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 768), (768, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (3072, 768), (768, 1))
    assert_size_stride(arg167_1, (3072, ), (1, ))
    assert_size_stride(arg168_1, (768, 3072), (3072, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, 768), (768, 1))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, 768), (768, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (3072, 768), (768, 1))
    assert_size_stride(arg179_1, (3072, ), (1, ))
    assert_size_stride(arg180_1, (768, 3072), (3072, 1))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (3072, 768), (768, 1))
    assert_size_stride(arg191_1, (3072, ), (1, ))
    assert_size_stride(arg192_1, (768, 3072), (3072, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(arg195_1, (4, 128), (128, 1))
    assert_size_stride(arg196_1, (4, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1, 128, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [mask], Original ATen: [aten.unsqueeze]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_unsqueeze_0.run(arg195_1, buf0, 65536, grid=grid(65536), stream=stream0)
        buf1 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf6 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_2, forward, forward_1, mean, mul, std, sub, truediv, x, x_4], Original ATen: [aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_embedding_mean_mul_std_sub_1.run(arg195_1, arg48_1, arg194_1, arg196_1, arg49_1, arg0_1, arg1_1, buf1, buf6, 512, 768, grid=grid(512), stream=stream0)
        del arg0_1
        del arg194_1
        del arg195_1
        del arg196_1
        del arg1_1
        del arg48_1
        del arg49_1
        buf7 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (512, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 768), (1, 768), 0), out=buf7)
        del arg50_1
        buf8 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (512, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 768), (1, 768), 0), out=buf8)
        del arg52_1
        buf9 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf7, arg51_1, buf9, 393216, grid=grid(393216), stream=stream0)
        del arg51_1
        buf10 = reinterpret_tensor(buf7, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf7  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf8, arg53_1, buf10, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg53_1
        buf11 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf10, (48, 64, 128), (8192, 128, 1), 0), out=buf11)
        buf15 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [eq, p_attn, scores, scores_1], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf11, buf15, 6144, 128, grid=grid(6144), stream=stream0)
        buf14 = reinterpret_tensor(buf9, (512, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (512, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), out=buf14)
        del arg54_1
        buf16 = reinterpret_tensor(buf6, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf14, arg55_1, buf16, 393216, grid=grid(393216), stream=stream0)
        del arg55_1
        buf17 = reinterpret_tensor(buf14, (48, 128, 64), (8192, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf16, (48, 128, 64), (8192, 64, 1), 0), out=buf17)
        buf18 = reinterpret_tensor(buf16, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf16  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf17, buf18, 393216, grid=grid(393216), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (512, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (512, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), out=buf19)
        del arg56_1
        buf24 = reinterpret_tensor(buf18, (4, 128, 768), (98304, 768, 1), 0); del buf18  # reuse
        # Source Nodes: [add_5, add_6, mean_2, mul_1, std_2, sub_1, truediv_2, x_7], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf1, buf19, arg57_1, arg2_1, arg3_1, buf24, 512, 768, grid=grid(512), stream=stream0)
        del arg2_1
        del arg3_1
        buf25 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (512, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 3072), (1, 768), 0), out=buf25)
        del arg58_1
        buf26 = reinterpret_tensor(buf25, (4, 128, 3072), (393216, 3072, 1), 0); del buf25  # reuse
        # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf26, arg59_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg59_1
        buf27 = reinterpret_tensor(buf24, (512, 768), (768, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg60_1, (3072, 768), (1, 3072), 0), out=buf27)
        del arg60_1
        buf28 = reinterpret_tensor(buf27, (4, 128, 768), (98304, 768, 1), 0); del buf27  # reuse
        buf33 = reinterpret_tensor(buf10, (4, 128, 768), (98304, 768, 1), 0); del buf10  # reuse
        # Source Nodes: [add_8, mean_4, mul_2, std_4, sub_2, truediv_3, x_12, x_7, x_8], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf28, buf1, buf19, arg57_1, arg61_1, arg4_1, arg5_1, buf33, 512, 768, grid=grid(512), stream=stream0)
        del arg4_1
        del arg57_1
        del arg5_1
        del arg61_1
        buf34 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (512, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 768), (1, 768), 0), out=buf34)
        del arg62_1
        buf35 = reinterpret_tensor(buf1, (512, 768), (768, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (512, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 768), (1, 768), 0), out=buf35)
        del arg64_1
        buf36 = reinterpret_tensor(buf8, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf34, arg63_1, buf36, 393216, grid=grid(393216), stream=stream0)
        del arg63_1
        buf37 = reinterpret_tensor(buf34, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf34  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf35, arg65_1, buf37, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg65_1
        buf38 = reinterpret_tensor(buf15, (48, 128, 128), (16384, 128, 1), 0); del buf15  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf37, (48, 64, 128), (8192, 128, 1), 0), out=buf38)
        buf42 = reinterpret_tensor(buf11, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf11  # reuse
        # Source Nodes: [eq_1, p_attn_2, scores_2, scores_3], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf38, buf42, 6144, 128, grid=grid(6144), stream=stream0)
        buf41 = reinterpret_tensor(buf37, (512, 768), (768, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (512, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 768), (1, 768), 0), out=buf41)
        del arg66_1
        buf43 = reinterpret_tensor(buf33, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf33  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf41, arg67_1, buf43, 393216, grid=grid(393216), stream=stream0)
        del arg67_1
        buf44 = reinterpret_tensor(buf41, (48, 128, 64), (8192, 64, 1), 0); del buf41  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf43, (48, 128, 64), (8192, 64, 1), 0), out=buf44)
        buf45 = reinterpret_tensor(buf43, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf43  # reuse
        # Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf44, buf45, 393216, grid=grid(393216), stream=stream0)
        buf46 = reinterpret_tensor(buf44, (512, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), out=buf46)
        del arg68_1
        buf51 = reinterpret_tensor(buf45, (4, 128, 768), (98304, 768, 1), 0); del buf45  # reuse
        # Source Nodes: [add_11, add_12, mean_6, mul_3, std_6, sub_3, truediv_5, x_15], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf28, buf46, arg69_1, arg6_1, arg7_1, buf51, 512, 768, grid=grid(512), stream=stream0)
        del arg6_1
        del arg7_1
        buf52 = reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 3072), (1, 768), 0), out=buf52)
        del arg70_1
        buf53 = reinterpret_tensor(buf52, (4, 128, 3072), (393216, 3072, 1), 0); del buf52  # reuse
        # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf53, arg71_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg71_1
        buf54 = reinterpret_tensor(buf51, (512, 768), (768, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg72_1, (3072, 768), (1, 3072), 0), out=buf54)
        del arg72_1
        buf55 = reinterpret_tensor(buf54, (4, 128, 768), (98304, 768, 1), 0); del buf54  # reuse
        buf60 = reinterpret_tensor(buf36, (4, 128, 768), (98304, 768, 1), 0); del buf36  # reuse
        # Source Nodes: [add_14, mean_8, mul_4, std_8, sub_4, truediv_6, x_15, x_16, x_20], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf55, buf28, buf46, arg69_1, arg73_1, arg8_1, arg9_1, buf60, 512, 768, grid=grid(512), stream=stream0)
        del arg69_1
        del arg73_1
        del arg8_1
        del arg9_1
        buf61 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (512, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), out=buf61)
        del arg74_1
        buf62 = reinterpret_tensor(buf28, (512, 768), (768, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (512, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), out=buf62)
        del arg76_1
        buf63 = reinterpret_tensor(buf35, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf61, arg75_1, buf63, 393216, grid=grid(393216), stream=stream0)
        del arg75_1
        buf64 = reinterpret_tensor(buf61, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf61  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf62, arg77_1, buf64, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg77_1
        buf65 = reinterpret_tensor(buf42, (48, 128, 128), (16384, 128, 1), 0); del buf42  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf64, (48, 64, 128), (8192, 128, 1), 0), out=buf65)
        buf69 = reinterpret_tensor(buf38, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf38  # reuse
        # Source Nodes: [eq_2, p_attn_4, scores_4, scores_5], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf65, buf69, 6144, 128, grid=grid(6144), stream=stream0)
        buf68 = reinterpret_tensor(buf64, (512, 768), (768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (512, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 768), (1, 768), 0), out=buf68)
        del arg78_1
        buf70 = reinterpret_tensor(buf60, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf60  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf68, arg79_1, buf70, 393216, grid=grid(393216), stream=stream0)
        del arg79_1
        buf71 = reinterpret_tensor(buf68, (48, 128, 64), (8192, 64, 1), 0); del buf68  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf70, (48, 128, 64), (8192, 64, 1), 0), out=buf71)
        buf72 = reinterpret_tensor(buf70, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf70  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf71, buf72, 393216, grid=grid(393216), stream=stream0)
        buf73 = reinterpret_tensor(buf71, (512, 768), (768, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (512, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 768), (1, 768), 0), out=buf73)
        del arg80_1
        buf78 = reinterpret_tensor(buf72, (4, 128, 768), (98304, 768, 1), 0); del buf72  # reuse
        # Source Nodes: [add_17, add_18, mean_10, mul_5, std_10, sub_5, truediv_8, x_23], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf55, buf73, arg81_1, arg10_1, arg11_1, buf78, 512, 768, grid=grid(512), stream=stream0)
        del arg10_1
        del arg11_1
        buf79 = reinterpret_tensor(buf53, (512, 3072), (3072, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 3072), (1, 768), 0), out=buf79)
        del arg82_1
        buf80 = reinterpret_tensor(buf79, (4, 128, 3072), (393216, 3072, 1), 0); del buf79  # reuse
        # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf80, arg83_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg83_1
        buf81 = reinterpret_tensor(buf78, (512, 768), (768, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg84_1, (3072, 768), (1, 3072), 0), out=buf81)
        del arg84_1
        buf82 = reinterpret_tensor(buf81, (4, 128, 768), (98304, 768, 1), 0); del buf81  # reuse
        buf87 = reinterpret_tensor(buf63, (4, 128, 768), (98304, 768, 1), 0); del buf63  # reuse
        # Source Nodes: [add_20, mean_12, mul_6, std_12, sub_6, truediv_9, x_23, x_24, x_28], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf82, buf55, buf73, arg81_1, arg85_1, arg12_1, arg13_1, buf87, 512, 768, grid=grid(512), stream=stream0)
        del arg12_1
        del arg13_1
        del arg81_1
        del arg85_1
        buf88 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), out=buf88)
        del arg86_1
        buf89 = reinterpret_tensor(buf55, (512, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), out=buf89)
        del arg88_1
        buf90 = reinterpret_tensor(buf62, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf62  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf88, arg87_1, buf90, 393216, grid=grid(393216), stream=stream0)
        del arg87_1
        buf91 = reinterpret_tensor(buf88, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf88  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf89, arg89_1, buf91, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg89_1
        buf92 = reinterpret_tensor(buf69, (48, 128, 128), (16384, 128, 1), 0); del buf69  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf91, (48, 64, 128), (8192, 128, 1), 0), out=buf92)
        buf96 = reinterpret_tensor(buf65, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf65  # reuse
        # Source Nodes: [eq_3, p_attn_6, scores_6, scores_7], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf92, buf96, 6144, 128, grid=grid(6144), stream=stream0)
        buf95 = reinterpret_tensor(buf91, (512, 768), (768, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), out=buf95)
        del arg90_1
        buf97 = reinterpret_tensor(buf87, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf87  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf95, arg91_1, buf97, 393216, grid=grid(393216), stream=stream0)
        del arg91_1
        buf98 = reinterpret_tensor(buf95, (48, 128, 64), (8192, 64, 1), 0); del buf95  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf97, (48, 128, 64), (8192, 64, 1), 0), out=buf98)
        buf99 = reinterpret_tensor(buf97, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf98, buf99, 393216, grid=grid(393216), stream=stream0)
        buf100 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), out=buf100)
        del arg92_1
        buf105 = reinterpret_tensor(buf99, (4, 128, 768), (98304, 768, 1), 0); del buf99  # reuse
        # Source Nodes: [add_23, add_24, mean_14, mul_7, std_14, sub_7, truediv_11, x_31], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf82, buf100, arg93_1, arg14_1, arg15_1, buf105, 512, 768, grid=grid(512), stream=stream0)
        del arg14_1
        del arg15_1
        buf106 = reinterpret_tensor(buf80, (512, 3072), (3072, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (512, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), out=buf106)
        del arg94_1
        buf107 = reinterpret_tensor(buf106, (4, 128, 3072), (393216, 3072, 1), 0); del buf106  # reuse
        # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf107, arg95_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg95_1
        buf108 = reinterpret_tensor(buf105, (512, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), out=buf108)
        del arg96_1
        buf109 = reinterpret_tensor(buf108, (4, 128, 768), (98304, 768, 1), 0); del buf108  # reuse
        buf114 = reinterpret_tensor(buf90, (4, 128, 768), (98304, 768, 1), 0); del buf90  # reuse
        # Source Nodes: [add_26, mean_16, mul_8, std_16, sub_8, truediv_12, x_31, x_32, x_36], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf109, buf82, buf100, arg93_1, arg97_1, arg16_1, arg17_1, buf114, 512, 768, grid=grid(512), stream=stream0)
        del arg16_1
        del arg17_1
        del arg93_1
        del arg97_1
        buf115 = reinterpret_tensor(buf82, (512, 768), (768, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (512, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 768), (1, 768), 0), out=buf115)
        del arg98_1
        buf116 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (512, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), out=buf116)
        del arg100_1
        buf117 = reinterpret_tensor(buf89, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf89  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf115, arg99_1, buf117, 393216, grid=grid(393216), stream=stream0)
        del arg99_1
        buf118 = reinterpret_tensor(buf115, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf115  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf116, arg101_1, buf118, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg101_1
        buf119 = reinterpret_tensor(buf96, (48, 128, 128), (16384, 128, 1), 0); del buf96  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf117, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf118, (48, 64, 128), (8192, 128, 1), 0), out=buf119)
        buf123 = reinterpret_tensor(buf92, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf92  # reuse
        # Source Nodes: [eq_4, p_attn_8, scores_8, scores_9], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf119, buf123, 6144, 128, grid=grid(6144), stream=stream0)
        buf122 = reinterpret_tensor(buf118, (512, 768), (768, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (512, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), out=buf122)
        del arg102_1
        buf124 = reinterpret_tensor(buf114, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf114  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf122, arg103_1, buf124, 393216, grid=grid(393216), stream=stream0)
        del arg103_1
        buf125 = reinterpret_tensor(buf122, (48, 128, 64), (8192, 64, 1), 0); del buf122  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf123, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf124, (48, 128, 64), (8192, 64, 1), 0), out=buf125)
        buf126 = reinterpret_tensor(buf124, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf125, buf126, 393216, grid=grid(393216), stream=stream0)
        buf127 = reinterpret_tensor(buf125, (512, 768), (768, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (512, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), out=buf127)
        del arg104_1
        buf132 = reinterpret_tensor(buf126, (4, 128, 768), (98304, 768, 1), 0); del buf126  # reuse
        # Source Nodes: [add_29, add_30, mean_18, mul_9, std_18, sub_9, truediv_14, x_39], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf109, buf127, arg105_1, arg18_1, arg19_1, buf132, 512, 768, grid=grid(512), stream=stream0)
        del arg18_1
        del arg19_1
        buf133 = reinterpret_tensor(buf107, (512, 3072), (3072, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (512, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 3072), (1, 768), 0), out=buf133)
        del arg106_1
        buf134 = reinterpret_tensor(buf133, (4, 128, 3072), (393216, 3072, 1), 0); del buf133  # reuse
        # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf134, arg107_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg107_1
        buf135 = reinterpret_tensor(buf132, (512, 768), (768, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf134, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg108_1, (3072, 768), (1, 3072), 0), out=buf135)
        del arg108_1
        buf136 = reinterpret_tensor(buf135, (4, 128, 768), (98304, 768, 1), 0); del buf135  # reuse
        buf141 = reinterpret_tensor(buf117, (4, 128, 768), (98304, 768, 1), 0); del buf117  # reuse
        # Source Nodes: [add_32, mean_20, mul_10, std_20, sub_10, truediv_15, x_39, x_40, x_44], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf136, buf109, buf127, arg105_1, arg109_1, arg20_1, arg21_1, buf141, 512, 768, grid=grid(512), stream=stream0)
        del arg105_1
        del arg109_1
        del arg20_1
        del arg21_1
        buf142 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (512, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 768), (1, 768), 0), out=buf142)
        del arg110_1
        buf143 = reinterpret_tensor(buf109, (512, 768), (768, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (512, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 768), (1, 768), 0), out=buf143)
        del arg112_1
        buf144 = reinterpret_tensor(buf116, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf116  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf142, arg111_1, buf144, 393216, grid=grid(393216), stream=stream0)
        del arg111_1
        buf145 = reinterpret_tensor(buf142, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf142  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf143, arg113_1, buf145, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg113_1
        buf146 = reinterpret_tensor(buf123, (48, 128, 128), (16384, 128, 1), 0); del buf123  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf145, (48, 64, 128), (8192, 128, 1), 0), out=buf146)
        buf150 = reinterpret_tensor(buf119, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf119  # reuse
        # Source Nodes: [eq_5, p_attn_10, scores_10, scores_11], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf146, buf150, 6144, 128, grid=grid(6144), stream=stream0)
        buf149 = reinterpret_tensor(buf145, (512, 768), (768, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (512, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 768), (1, 768), 0), out=buf149)
        del arg114_1
        buf151 = reinterpret_tensor(buf141, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf141  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf149, arg115_1, buf151, 393216, grid=grid(393216), stream=stream0)
        del arg115_1
        buf152 = reinterpret_tensor(buf149, (48, 128, 64), (8192, 64, 1), 0); del buf149  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf150, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf151, (48, 128, 64), (8192, 64, 1), 0), out=buf152)
        buf153 = reinterpret_tensor(buf151, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf151  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf152, buf153, 393216, grid=grid(393216), stream=stream0)
        buf154 = reinterpret_tensor(buf152, (512, 768), (768, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), out=buf154)
        del arg116_1
        buf159 = reinterpret_tensor(buf153, (4, 128, 768), (98304, 768, 1), 0); del buf153  # reuse
        # Source Nodes: [add_35, add_36, mean_22, mul_11, std_22, sub_11, truediv_17, x_47], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf136, buf154, arg117_1, arg22_1, arg23_1, buf159, 512, 768, grid=grid(512), stream=stream0)
        del arg22_1
        del arg23_1
        buf160 = reinterpret_tensor(buf134, (512, 3072), (3072, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (512, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 3072), (1, 768), 0), out=buf160)
        del arg118_1
        buf161 = reinterpret_tensor(buf160, (4, 128, 3072), (393216, 3072, 1), 0); del buf160  # reuse
        # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf161, arg119_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg119_1
        buf162 = reinterpret_tensor(buf159, (512, 768), (768, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg120_1, (3072, 768), (1, 3072), 0), out=buf162)
        del arg120_1
        buf163 = reinterpret_tensor(buf162, (4, 128, 768), (98304, 768, 1), 0); del buf162  # reuse
        buf168 = reinterpret_tensor(buf144, (4, 128, 768), (98304, 768, 1), 0); del buf144  # reuse
        # Source Nodes: [add_38, mean_24, mul_12, std_24, sub_12, truediv_18, x_47, x_48, x_52], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf163, buf136, buf154, arg117_1, arg121_1, arg24_1, arg25_1, buf168, 512, 768, grid=grid(512), stream=stream0)
        del arg117_1
        del arg121_1
        del arg24_1
        del arg25_1
        buf169 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (512, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), out=buf169)
        del arg122_1
        buf170 = reinterpret_tensor(buf136, (512, 768), (768, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (512, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 768), (1, 768), 0), out=buf170)
        del arg124_1
        buf171 = reinterpret_tensor(buf143, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf143  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf169, arg123_1, buf171, 393216, grid=grid(393216), stream=stream0)
        del arg123_1
        buf172 = reinterpret_tensor(buf169, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf169  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf170, arg125_1, buf172, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg125_1
        buf173 = reinterpret_tensor(buf150, (48, 128, 128), (16384, 128, 1), 0); del buf150  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf172, (48, 64, 128), (8192, 128, 1), 0), out=buf173)
        buf177 = reinterpret_tensor(buf146, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf146  # reuse
        # Source Nodes: [eq_6, p_attn_12, scores_12, scores_13], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf173, buf177, 6144, 128, grid=grid(6144), stream=stream0)
        buf176 = reinterpret_tensor(buf172, (512, 768), (768, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (512, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 768), (1, 768), 0), out=buf176)
        del arg126_1
        buf178 = reinterpret_tensor(buf168, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf168  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf176, arg127_1, buf178, 393216, grid=grid(393216), stream=stream0)
        del arg127_1
        buf179 = reinterpret_tensor(buf176, (48, 128, 64), (8192, 64, 1), 0); del buf176  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf177, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf178, (48, 128, 64), (8192, 64, 1), 0), out=buf179)
        buf180 = reinterpret_tensor(buf178, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf178  # reuse
        # Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf179, buf180, 393216, grid=grid(393216), stream=stream0)
        buf181 = reinterpret_tensor(buf179, (512, 768), (768, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (512, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), out=buf181)
        del arg128_1
        buf186 = reinterpret_tensor(buf180, (4, 128, 768), (98304, 768, 1), 0); del buf180  # reuse
        # Source Nodes: [add_41, add_42, mean_26, mul_13, std_26, sub_13, truediv_20, x_55], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf163, buf181, arg129_1, arg26_1, arg27_1, buf186, 512, 768, grid=grid(512), stream=stream0)
        del arg26_1
        del arg27_1
        buf187 = reinterpret_tensor(buf161, (512, 3072), (3072, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (512, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), out=buf187)
        del arg130_1
        buf188 = reinterpret_tensor(buf187, (4, 128, 3072), (393216, 3072, 1), 0); del buf187  # reuse
        # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf188, arg131_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg131_1
        buf189 = reinterpret_tensor(buf186, (512, 768), (768, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), out=buf189)
        del arg132_1
        buf190 = reinterpret_tensor(buf189, (4, 128, 768), (98304, 768, 1), 0); del buf189  # reuse
        buf195 = reinterpret_tensor(buf171, (4, 128, 768), (98304, 768, 1), 0); del buf171  # reuse
        # Source Nodes: [add_44, mean_28, mul_14, std_28, sub_14, truediv_21, x_55, x_56, x_60], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf190, buf163, buf181, arg129_1, arg133_1, arg28_1, arg29_1, buf195, 512, 768, grid=grid(512), stream=stream0)
        del arg129_1
        del arg133_1
        del arg28_1
        del arg29_1
        buf196 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), out=buf196)
        del arg134_1
        buf197 = reinterpret_tensor(buf163, (512, 768), (768, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), out=buf197)
        del arg136_1
        buf198 = reinterpret_tensor(buf170, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf196, arg135_1, buf198, 393216, grid=grid(393216), stream=stream0)
        del arg135_1
        buf199 = reinterpret_tensor(buf196, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf196  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf197, arg137_1, buf199, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg137_1
        buf200 = reinterpret_tensor(buf177, (48, 128, 128), (16384, 128, 1), 0); del buf177  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf199, (48, 64, 128), (8192, 128, 1), 0), out=buf200)
        buf204 = reinterpret_tensor(buf173, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf173  # reuse
        # Source Nodes: [eq_7, p_attn_14, scores_14, scores_15], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf200, buf204, 6144, 128, grid=grid(6144), stream=stream0)
        buf203 = reinterpret_tensor(buf199, (512, 768), (768, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), out=buf203)
        del arg138_1
        buf205 = reinterpret_tensor(buf195, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf195  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf203, arg139_1, buf205, 393216, grid=grid(393216), stream=stream0)
        del arg139_1
        buf206 = reinterpret_tensor(buf203, (48, 128, 64), (8192, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf205, (48, 128, 64), (8192, 64, 1), 0), out=buf206)
        buf207 = reinterpret_tensor(buf205, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf205  # reuse
        # Source Nodes: [contiguous_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf206, buf207, 393216, grid=grid(393216), stream=stream0)
        buf208 = reinterpret_tensor(buf206, (512, 768), (768, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), out=buf208)
        del arg140_1
        buf213 = reinterpret_tensor(buf207, (4, 128, 768), (98304, 768, 1), 0); del buf207  # reuse
        # Source Nodes: [add_47, add_48, mean_30, mul_15, std_30, sub_15, truediv_23, x_63], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf190, buf208, arg141_1, arg30_1, arg31_1, buf213, 512, 768, grid=grid(512), stream=stream0)
        del arg30_1
        del arg31_1
        buf214 = reinterpret_tensor(buf188, (512, 3072), (3072, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf213, (512, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), out=buf214)
        del arg142_1
        buf215 = reinterpret_tensor(buf214, (4, 128, 3072), (393216, 3072, 1), 0); del buf214  # reuse
        # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf215, arg143_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg143_1
        buf216 = reinterpret_tensor(buf213, (512, 768), (768, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), out=buf216)
        del arg144_1
        buf217 = reinterpret_tensor(buf216, (4, 128, 768), (98304, 768, 1), 0); del buf216  # reuse
        buf222 = reinterpret_tensor(buf198, (4, 128, 768), (98304, 768, 1), 0); del buf198  # reuse
        # Source Nodes: [add_50, mean_32, mul_16, std_32, sub_16, truediv_24, x_63, x_64, x_68], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf217, buf190, buf208, arg141_1, arg145_1, arg32_1, arg33_1, buf222, 512, 768, grid=grid(512), stream=stream0)
        del arg141_1
        del arg145_1
        del arg32_1
        del arg33_1
        buf223 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (512, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 768), (1, 768), 0), out=buf223)
        del arg146_1
        buf224 = reinterpret_tensor(buf190, (512, 768), (768, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (512, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 768), (1, 768), 0), out=buf224)
        del arg148_1
        buf225 = reinterpret_tensor(buf197, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf223, arg147_1, buf225, 393216, grid=grid(393216), stream=stream0)
        del arg147_1
        buf226 = reinterpret_tensor(buf223, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf223  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf224, arg149_1, buf226, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg149_1
        buf227 = reinterpret_tensor(buf204, (48, 128, 128), (16384, 128, 1), 0); del buf204  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf226, (48, 64, 128), (8192, 128, 1), 0), out=buf227)
        buf231 = reinterpret_tensor(buf200, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf200  # reuse
        # Source Nodes: [eq_8, p_attn_16, scores_16, scores_17], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf227, buf231, 6144, 128, grid=grid(6144), stream=stream0)
        buf230 = reinterpret_tensor(buf226, (512, 768), (768, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (512, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 768), (1, 768), 0), out=buf230)
        del arg150_1
        buf232 = reinterpret_tensor(buf222, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf222  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf230, arg151_1, buf232, 393216, grid=grid(393216), stream=stream0)
        del arg151_1
        buf233 = reinterpret_tensor(buf230, (48, 128, 64), (8192, 64, 1), 0); del buf230  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf232, (48, 128, 64), (8192, 64, 1), 0), out=buf233)
        buf234 = reinterpret_tensor(buf232, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf232  # reuse
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf233, buf234, 393216, grid=grid(393216), stream=stream0)
        buf235 = reinterpret_tensor(buf233, (512, 768), (768, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (512, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), out=buf235)
        del arg152_1
        buf240 = reinterpret_tensor(buf234, (4, 128, 768), (98304, 768, 1), 0); del buf234  # reuse
        # Source Nodes: [add_53, add_54, mean_34, mul_17, std_34, sub_17, truediv_26, x_71], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf217, buf235, arg153_1, arg34_1, arg35_1, buf240, 512, 768, grid=grid(512), stream=stream0)
        del arg34_1
        del arg35_1
        buf241 = reinterpret_tensor(buf215, (512, 3072), (3072, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (512, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 3072), (1, 768), 0), out=buf241)
        del arg154_1
        buf242 = reinterpret_tensor(buf241, (4, 128, 3072), (393216, 3072, 1), 0); del buf241  # reuse
        # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf242, arg155_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg155_1
        buf243 = reinterpret_tensor(buf240, (512, 768), (768, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg156_1, (3072, 768), (1, 3072), 0), out=buf243)
        del arg156_1
        buf244 = reinterpret_tensor(buf243, (4, 128, 768), (98304, 768, 1), 0); del buf243  # reuse
        buf249 = reinterpret_tensor(buf225, (4, 128, 768), (98304, 768, 1), 0); del buf225  # reuse
        # Source Nodes: [add_56, mean_36, mul_18, std_36, sub_18, truediv_27, x_71, x_72, x_76], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf244, buf217, buf235, arg153_1, arg157_1, arg36_1, arg37_1, buf249, 512, 768, grid=grid(512), stream=stream0)
        del arg153_1
        del arg157_1
        del arg36_1
        del arg37_1
        buf250 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), out=buf250)
        del arg158_1
        buf251 = reinterpret_tensor(buf217, (512, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 768), (1, 768), 0), out=buf251)
        del arg160_1
        buf252 = reinterpret_tensor(buf224, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf224  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf250, arg159_1, buf252, 393216, grid=grid(393216), stream=stream0)
        del arg159_1
        buf253 = reinterpret_tensor(buf250, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf250  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf251, arg161_1, buf253, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg161_1
        buf254 = reinterpret_tensor(buf231, (48, 128, 128), (16384, 128, 1), 0); del buf231  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf253, (48, 64, 128), (8192, 128, 1), 0), out=buf254)
        buf258 = reinterpret_tensor(buf227, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf227  # reuse
        # Source Nodes: [eq_9, p_attn_18, scores_18, scores_19], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf254, buf258, 6144, 128, grid=grid(6144), stream=stream0)
        buf257 = reinterpret_tensor(buf253, (512, 768), (768, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 768), (1, 768), 0), out=buf257)
        del arg162_1
        buf259 = reinterpret_tensor(buf249, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf249  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf257, arg163_1, buf259, 393216, grid=grid(393216), stream=stream0)
        del arg163_1
        buf260 = reinterpret_tensor(buf257, (48, 128, 64), (8192, 64, 1), 0); del buf257  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf258, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf259, (48, 128, 64), (8192, 64, 1), 0), out=buf260)
        buf261 = reinterpret_tensor(buf259, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf260, buf261, 393216, grid=grid(393216), stream=stream0)
        buf262 = reinterpret_tensor(buf260, (512, 768), (768, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf261, (512, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 768), (1, 768), 0), out=buf262)
        del arg164_1
        buf267 = reinterpret_tensor(buf261, (4, 128, 768), (98304, 768, 1), 0); del buf261  # reuse
        # Source Nodes: [add_59, add_60, mean_38, mul_19, std_38, sub_19, truediv_29, x_79], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf244, buf262, arg165_1, arg38_1, arg39_1, buf267, 512, 768, grid=grid(512), stream=stream0)
        del arg38_1
        del arg39_1
        buf268 = reinterpret_tensor(buf242, (512, 3072), (3072, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (512, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 3072), (1, 768), 0), out=buf268)
        del arg166_1
        buf269 = reinterpret_tensor(buf268, (4, 128, 3072), (393216, 3072, 1), 0); del buf268  # reuse
        # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf269, arg167_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg167_1
        buf270 = reinterpret_tensor(buf267, (512, 768), (768, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg168_1, (3072, 768), (1, 3072), 0), out=buf270)
        del arg168_1
        buf271 = reinterpret_tensor(buf270, (4, 128, 768), (98304, 768, 1), 0); del buf270  # reuse
        buf276 = reinterpret_tensor(buf252, (4, 128, 768), (98304, 768, 1), 0); del buf252  # reuse
        # Source Nodes: [add_62, mean_40, mul_20, std_40, sub_20, truediv_30, x_79, x_80, x_84], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf271, buf244, buf262, arg165_1, arg169_1, arg40_1, arg41_1, buf276, 512, 768, grid=grid(512), stream=stream0)
        del arg165_1
        del arg169_1
        del arg40_1
        del arg41_1
        buf277 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), out=buf277)
        del arg170_1
        buf278 = reinterpret_tensor(buf244, (512, 768), (768, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), out=buf278)
        del arg172_1
        buf279 = reinterpret_tensor(buf251, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf251  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf277, arg171_1, buf279, 393216, grid=grid(393216), stream=stream0)
        del arg171_1
        buf280 = reinterpret_tensor(buf277, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf277  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf278, arg173_1, buf280, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg173_1
        buf281 = reinterpret_tensor(buf258, (48, 128, 128), (16384, 128, 1), 0); del buf258  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf280, (48, 64, 128), (8192, 128, 1), 0), out=buf281)
        buf285 = reinterpret_tensor(buf254, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf254  # reuse
        # Source Nodes: [eq_10, p_attn_20, scores_20, scores_21], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf281, buf285, 6144, 128, grid=grid(6144), stream=stream0)
        buf284 = reinterpret_tensor(buf280, (512, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 768), (1, 768), 0), out=buf284)
        del arg174_1
        buf286 = reinterpret_tensor(buf276, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf276  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf284, arg175_1, buf286, 393216, grid=grid(393216), stream=stream0)
        del arg175_1
        buf287 = reinterpret_tensor(buf284, (48, 128, 64), (8192, 64, 1), 0); del buf284  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf285, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf286, (48, 128, 64), (8192, 64, 1), 0), out=buf287)
        buf288 = reinterpret_tensor(buf286, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf286  # reuse
        # Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf287, buf288, 393216, grid=grid(393216), stream=stream0)
        buf289 = reinterpret_tensor(buf287, (512, 768), (768, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf288, (512, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 768), (1, 768), 0), out=buf289)
        del arg176_1
        buf294 = reinterpret_tensor(buf288, (4, 128, 768), (98304, 768, 1), 0); del buf288  # reuse
        # Source Nodes: [add_65, add_66, mean_42, mul_21, std_42, sub_21, truediv_32, x_87], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf271, buf289, arg177_1, arg42_1, arg43_1, buf294, 512, 768, grid=grid(512), stream=stream0)
        del arg42_1
        del arg43_1
        buf295 = reinterpret_tensor(buf269, (512, 3072), (3072, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (512, 768), (768, 1), 0), reinterpret_tensor(arg178_1, (768, 3072), (1, 768), 0), out=buf295)
        del arg178_1
        buf296 = reinterpret_tensor(buf295, (4, 128, 3072), (393216, 3072, 1), 0); del buf295  # reuse
        # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf296, arg179_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg179_1
        buf297 = reinterpret_tensor(buf294, (512, 768), (768, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg180_1, (3072, 768), (1, 3072), 0), out=buf297)
        del arg180_1
        buf298 = reinterpret_tensor(buf297, (4, 128, 768), (98304, 768, 1), 0); del buf297  # reuse
        buf303 = reinterpret_tensor(buf279, (4, 128, 768), (98304, 768, 1), 0); del buf279  # reuse
        # Source Nodes: [add_68, mean_44, mul_22, std_44, sub_22, truediv_33, x_87, x_88, x_92], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_8.run(buf298, buf271, buf289, arg177_1, arg181_1, arg44_1, arg45_1, buf303, 512, 768, grid=grid(512), stream=stream0)
        del arg177_1
        del arg181_1
        del arg44_1
        del arg45_1
        buf304 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), out=buf304)
        del arg182_1
        buf305 = reinterpret_tensor(buf271, (512, 768), (768, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), out=buf305)
        del arg184_1
        buf306 = reinterpret_tensor(buf278, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf278  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf304, arg183_1, buf306, 393216, grid=grid(393216), stream=stream0)
        del arg183_1
        buf307 = reinterpret_tensor(buf304, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf304  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf305, arg185_1, buf307, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del arg185_1
        del buf305
        buf308 = reinterpret_tensor(buf285, (48, 128, 128), (16384, 128, 1), 0); del buf285  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf306, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf307, (48, 64, 128), (8192, 128, 1), 0), out=buf308)
        del buf306
        buf312 = reinterpret_tensor(buf281, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf281  # reuse
        # Source Nodes: [eq_11, p_attn_22, scores_22, scores_23], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_div_eq_masked_fill_4.run(buf0, buf308, buf312, 6144, 128, grid=grid(6144), stream=stream0)
        del buf308
        buf311 = reinterpret_tensor(buf307, (512, 768), (768, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), out=buf311)
        del arg186_1
        buf313 = reinterpret_tensor(buf303, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf303  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf311, arg187_1, buf313, 393216, grid=grid(393216), stream=stream0)
        del arg187_1
        buf314 = reinterpret_tensor(buf311, (48, 128, 64), (8192, 64, 1), 0); del buf311  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf312, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf313, (48, 128, 64), (8192, 64, 1), 0), out=buf314)
        del buf312
        buf315 = reinterpret_tensor(buf313, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf313  # reuse
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf314, buf315, 393216, grid=grid(393216), stream=stream0)
        buf316 = reinterpret_tensor(buf314, (512, 768), (768, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), out=buf316)
        del arg188_1
        buf321 = reinterpret_tensor(buf315, (4, 128, 768), (98304, 768, 1), 0); del buf315  # reuse
        # Source Nodes: [add_71, add_72, mean_46, mul_23, std_46, sub_23, truediv_35, x_95], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
        triton_per_fused_add_div_mean_mul_std_sub_6.run(buf298, buf316, arg189_1, arg46_1, arg47_1, buf321, 512, 768, grid=grid(512), stream=stream0)
        del arg46_1
        del arg47_1
        buf322 = reinterpret_tensor(buf296, (512, 3072), (3072, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf321, (512, 768), (768, 1), 0), reinterpret_tensor(arg190_1, (768, 3072), (1, 768), 0), out=buf322)
        del arg190_1
        buf323 = reinterpret_tensor(buf322, (4, 128, 3072), (393216, 3072, 1), 0); del buf322  # reuse
        # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_activation], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf323, arg191_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg191_1
        buf324 = reinterpret_tensor(buf321, (512, 768), (768, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg192_1, (3072, 768), (1, 3072), 0), out=buf324)
        del arg192_1
        del buf323
        buf325 = reinterpret_tensor(buf324, (4, 128, 768), (98304, 768, 1), 0); del buf324  # reuse
        # Source Nodes: [x_95, x_96], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf325, buf298, buf316, arg189_1, arg193_1, 393216, grid=grid(393216), stream=stream0)
        del arg189_1
        del arg193_1
        return (buf325, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((20005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg196_1 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BERT_pytorch', benchmark_compiled_module)
