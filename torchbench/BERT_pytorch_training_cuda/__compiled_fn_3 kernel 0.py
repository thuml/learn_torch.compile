
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


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kqzas6lu7ahpwrpvawbujebnct4yxd5mg7lhw542pgxlway6gi.py
# Source Nodes: [add, add_2, forward, forward_1, l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0, mean, mul, std, sub, truediv, x, x_4], Original ATen: [aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
# add => add
# add_2 => add_2
# forward => embedding
# forward_1 => embedding_1
# l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0 => view
# mean => mean
# mul => mul
# std => sqrt, var
# sub => sub
# truediv => div
# x => add_1
# x_4 => add_3
triton_per_fused_add_div_embedding_mean_mul_std_sub_view_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_mean_mul_std_sub_view_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp37 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp31 = 767.0
    tmp32 = tmp30 / tmp31
    tmp33 = tl.sqrt(tmp32)
    tmp34 = 768.0
    tmp35 = tmp16 / tmp34
    tmp36 = tmp12 - tmp35
    tmp38 = tmp37 * tmp36
    tmp39 = 1e-06
    tmp40 = tmp33 + tmp39
    tmp41 = tmp38 / tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp12, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp33, xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp36, rmask & xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhvlquuctgffntxlapedvaug3bm6awfwtg6bg3lgy7akjqt3zie.py
# Source Nodes: [attn, eq, p_attn, scores, scores_1], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
# attn => clone_3
# eq => eq
# p_attn => amax, div_2, exp, sub_1, sum_1
# scores => div_1
# scores_1 => full_default, where
triton_per_fused__softmax_clone_div_eq_masked_fill_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_div_eq_masked_fill_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4), tmp12, None)
    tl.store(out_ptr1 + (x4), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmrongejkwrr2x5wtk7inrurnr2xclan6beeug5imghnrxdb3dg.py
# Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_output_linear], Original ATen: [aten.view]
# l__mod___transformer_blocks_0_lambda_module_attention_output_linear => view_16
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 128)) + (8192*(x0 // 64)) + (98304*(x1 // 128)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caechaurtpxqvjpkwk2tpn7k3ncyvmnthf2lvtruc5uoh6bsfnq7.py
# Source Nodes: [add_5, add_6, l__mod___transformer_blocks_0_feed_forward_w_1, mean_2, mul_1, std_2, sub_1, truediv_2, x_7], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
# add_5 => add_5
# add_6 => add_6
# l__mod___transformer_blocks_0_feed_forward_w_1 => view_18
# mean_2 => mean_1
# mul_1 => mul_1
# std_2 => sqrt_1, var_1
# sub_1 => sub_2
# truediv_2 => div_3
# x_7 => add_4
triton_per_fused_add_div_mean_mul_std_sub_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_std_sub_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp29 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp23 = 767.0
    tmp24 = tmp22 / tmp23
    tmp25 = tl.sqrt(tmp24)
    tmp26 = 768.0
    tmp27 = tmp8 / tmp26
    tmp28 = tmp4 - tmp27
    tmp30 = tmp29 * tmp28
    tmp31 = 1e-06
    tmp32 = tmp25 + tmp31
    tmp33 = tmp30 / tmp32
    tmp35 = tmp33 + tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, xmask)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp28, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfoosfz5satsq5nss6psnggs2oj6hojkvn5jgxzmnmb3nuk37oan.py
# Source Nodes: [l__mod___transformer_blocks_0_feed_forward_activation, l__mod___transformer_blocks_0_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
# l__mod___transformer_blocks_0_feed_forward_activation => add_7, erf, mul_2, mul_3, mul_4
# l__mod___transformer_blocks_0_feed_forward_w_2 => view_20
triton_poi_fused_gelu_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/5o/c5of3g4demg6g3e5xvipske7xewgfcs6hlra2aup36ivie5hundu.py
# Source Nodes: [add_8, l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0, mean_4, mul_2, std_4, sub_2, truediv_3, x_12, x_7, x_8], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
# add_8 => add_9
# l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0 => view_22
# mean_4 => mean_2
# mul_2 => mul_5
# std_4 => sqrt_2, var_2
# sub_2 => sub_3
# truediv_3 => div_4
# x_12 => add_10
# x_7 => add_4
# x_8 => add_8
triton_per_fused_add_div_mean_mul_std_sub_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_std_sub_view_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp33 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = 767.0
    tmp28 = tmp26 / tmp27
    tmp29 = tl.sqrt(tmp28)
    tmp30 = 768.0
    tmp31 = tmp12 / tmp30
    tmp32 = tmp8 - tmp31
    tmp34 = tmp33 * tmp32
    tmp35 = 1e-06
    tmp36 = tmp29 + tmp35
    tmp37 = tmp34 / tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, xmask)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdj2nugunslhite22gth4mpejvh2kby63sw43kgxnueascrq4dj.py
# Source Nodes: [attn_10, eq, p_attn, p_attn_10, p_attn_12, p_attn_14, p_attn_16, p_attn_18, p_attn_2, p_attn_20, p_attn_4, p_attn_6, p_attn_8, scores, scores_1, scores_10, scores_11, scores_12, scores_13, scores_14, scores_15, scores_16, scores_17, scores_18, scores_19, scores_2, scores_20, scores_21, scores_3, scores_4, scores_5, scores_6, scores_7, scores_8, scores_9], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.div, aten.eq, aten.masked_fill]
# attn_10 => clone_93
# eq => eq
# p_attn => div_2, exp, sub_1
# p_attn_10 => div_22, exp_5, sub_16
# p_attn_12 => div_26, exp_6, sub_19
# p_attn_14 => div_30, exp_7, sub_22
# p_attn_16 => div_34, exp_8, sub_25
# p_attn_18 => div_38, exp_9, sub_28
# p_attn_2 => div_6, exp_1, sub_4
# p_attn_20 => amax_10, div_42, exp_10, sub_31, sum_11
# p_attn_4 => div_10, exp_2, sub_7
# p_attn_6 => div_14, exp_3, sub_10
# p_attn_8 => div_18, exp_4, sub_13
# scores => div_1
# scores_1 => full_default, where
# scores_10 => div_21
# scores_11 => where_5
# scores_12 => div_25
# scores_13 => where_6
# scores_14 => div_29
# scores_15 => where_7
# scores_16 => div_33
# scores_17 => where_8
# scores_18 => div_37
# scores_19 => where_9
# scores_2 => div_5
# scores_20 => div_41
# scores_21 => where_10
# scores_3 => where_1
# scores_4 => div_9
# scores_5 => where_2
# scores_6 => div_13
# scores_7 => where_3
# scores_8 => div_17
# scores_9 => where_4
triton_per_fused__softmax_clone_detach_div_eq_masked_fill_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: 'i32', 35: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(34, 35))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_div_eq_masked_fill_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp20 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_out_ptr1 + (r3 + (128*x4)), rmask, other=0.0)
    tmp31 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_out_ptr2 + (r3 + (128*x4)), rmask, other=0.0)
    tmp39 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr7 + (x4), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_out_ptr3 + (r3 + (128*x4)), rmask, other=0.0)
    tmp47 = tl.load(in_ptr8 + (x4), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr9 + (x4), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_out_ptr4 + (r3 + (128*x4)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr10 + (x4), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_out_ptr5 + (r3 + (128*x4)), rmask, other=0.0)
    tmp63 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr13 + (x4), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_out_ptr6 + (r3 + (128*x4)), rmask, other=0.0)
    tmp71 = tl.load(in_ptr14 + (x4), None, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr15 + (x4), None, eviction_policy='evict_last')
    tmp76 = tl.load(in_out_ptr7 + (r3 + (128*x4)), rmask, other=0.0)
    tmp79 = tl.load(in_ptr16 + (x4), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr17 + (x4), None, eviction_policy='evict_last')
    tmp84 = tl.load(in_out_ptr8 + (r3 + (128*x4)), rmask, other=0.0)
    tmp87 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr19 + (x4), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_out_ptr9 + (r3 + (128*x4)), rmask, other=0.0)
    tmp95 = tl.load(in_ptr20 + (x4), None, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr21 + (x4), None, eviction_policy='evict_last')
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
    tmp21 = tmp20 / tmp5
    tmp22 = tl.where(tmp3, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp25 = tl.exp(tmp24)
    tmp27 = tmp25 / tmp26
    tmp29 = tmp28 / tmp5
    tmp30 = tl.where(tmp3, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp33 = tl.exp(tmp32)
    tmp35 = tmp33 / tmp34
    tmp37 = tmp36 / tmp5
    tmp38 = tl.where(tmp3, tmp7, tmp37)
    tmp40 = tmp38 - tmp39
    tmp41 = tl.exp(tmp40)
    tmp43 = tmp41 / tmp42
    tmp45 = tmp44 / tmp5
    tmp46 = tl.where(tmp3, tmp7, tmp45)
    tmp48 = tmp46 - tmp47
    tmp49 = tl.exp(tmp48)
    tmp51 = tmp49 / tmp50
    tmp53 = tmp52 / tmp5
    tmp54 = tl.where(tmp3, tmp7, tmp53)
    tmp56 = tmp54 - tmp55
    tmp57 = tl.exp(tmp56)
    tmp59 = tmp57 / tmp58
    tmp61 = tmp60 / tmp5
    tmp62 = tl.where(tmp3, tmp7, tmp61)
    tmp64 = tmp62 - tmp63
    tmp65 = tl.exp(tmp64)
    tmp67 = tmp65 / tmp66
    tmp69 = tmp68 / tmp5
    tmp70 = tl.where(tmp3, tmp7, tmp69)
    tmp72 = tmp70 - tmp71
    tmp73 = tl.exp(tmp72)
    tmp75 = tmp73 / tmp74
    tmp77 = tmp76 / tmp5
    tmp78 = tl.where(tmp3, tmp7, tmp77)
    tmp80 = tmp78 - tmp79
    tmp81 = tl.exp(tmp80)
    tmp83 = tmp81 / tmp82
    tmp85 = tmp84 / tmp5
    tmp86 = tl.where(tmp3, tmp7, tmp85)
    tmp88 = tmp86 - tmp87
    tmp89 = tl.exp(tmp88)
    tmp91 = tmp89 / tmp90
    tmp93 = tmp92 / tmp5
    tmp94 = tl.where(tmp3, tmp7, tmp93)
    tmp96 = tmp94 - tmp95
    tmp97 = tl.exp(tmp96)
    tmp99 = tmp97 / tmp98
    tl.store(out_ptr2 + (r3 + (128*x4)), tmp19, rmask)
    tl.store(out_ptr3 + (r3 + (128*x4)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r3 + (128*x4)), tmp27, rmask)
    tl.store(in_out_ptr1 + (r3 + (128*x4)), tmp35, rmask)
    tl.store(in_out_ptr2 + (r3 + (128*x4)), tmp43, rmask)
    tl.store(in_out_ptr3 + (r3 + (128*x4)), tmp51, rmask)
    tl.store(in_out_ptr4 + (r3 + (128*x4)), tmp59, rmask)
    tl.store(in_out_ptr5 + (r3 + (128*x4)), tmp67, rmask)
    tl.store(in_out_ptr6 + (r3 + (128*x4)), tmp75, rmask)
    tl.store(in_out_ptr7 + (r3 + (128*x4)), tmp83, rmask)
    tl.store(in_out_ptr8 + (r3 + (128*x4)), tmp91, rmask)
    tl.store(in_out_ptr9 + (r3 + (128*x4)), tmp99, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggvcpriqndynclcftfbufcoyf22cfulx5s5r5e3djlej3yw35sv.py
# Source Nodes: [attn_11, eq, p_attn_22, scores_1, scores_22, scores_23], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.div, aten.eq, aten.masked_fill]
# attn_11 => clone_102
# eq => eq
# p_attn_22 => amax_11, div_46, exp_11, sub_34, sum_12
# scores_1 => full_default
# scores_22 => div_45
# scores_23 => where_11
triton_per_fused__softmax_clone_detach_div_eq_masked_fill_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_div_eq_masked_fill_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr3 + (r3 + (128*x4)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfoctwismeue7soga5k7rrhvpqbhdh2n3xzfdnnglss3i2qxl2f.py
# Source Nodes: [x_95, x_96], Original ATen: [aten.add]
# x_95 => add_81
# x_96 => add_85
triton_poi_fused_add_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': ['in_out_ptr0']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (20005, 768), (768, 1))
    assert_size_stride(primals_50, (3, 768), (768, 1))
    assert_size_stride(primals_51, (768, 768), (768, 1))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 768), (768, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, 768), (768, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768), (768, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (3072, 768), (768, 1))
    assert_size_stride(primals_60, (3072, ), (1, ))
    assert_size_stride(primals_61, (768, 3072), (3072, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, 768), (768, 1))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768), (768, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, 768), (768, 1))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (3072, 768), (768, 1))
    assert_size_stride(primals_72, (3072, ), (1, ))
    assert_size_stride(primals_73, (768, 3072), (3072, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, 768), (768, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, 768), (768, 1))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, 768), (768, 1))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768), (768, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (3072, 768), (768, 1))
    assert_size_stride(primals_84, (3072, ), (1, ))
    assert_size_stride(primals_85, (768, 3072), (3072, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, 768), (768, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768), (768, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, 768), (768, 1))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, 768), (768, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (3072, 768), (768, 1))
    assert_size_stride(primals_96, (3072, ), (1, ))
    assert_size_stride(primals_97, (768, 3072), (3072, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 768), (768, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, 768), (768, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, 768), (768, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (3072, 768), (768, 1))
    assert_size_stride(primals_108, (3072, ), (1, ))
    assert_size_stride(primals_109, (768, 3072), (3072, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, 768), (768, 1))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768), (768, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, 768), (768, 1))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (3072, 768), (768, 1))
    assert_size_stride(primals_120, (3072, ), (1, ))
    assert_size_stride(primals_121, (768, 3072), (3072, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, 768), (768, 1))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 768), (768, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, 768), (768, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (3072, 768), (768, 1))
    assert_size_stride(primals_132, (3072, ), (1, ))
    assert_size_stride(primals_133, (768, 3072), (3072, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, 768), (768, 1))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, 768), (768, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, 768), (768, 1))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 768), (768, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (3072, 768), (768, 1))
    assert_size_stride(primals_144, (3072, ), (1, ))
    assert_size_stride(primals_145, (768, 3072), (3072, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, 768), (768, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 768), (768, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, 768), (768, 1))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, 768), (768, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (3072, 768), (768, 1))
    assert_size_stride(primals_156, (3072, ), (1, ))
    assert_size_stride(primals_157, (768, 3072), (3072, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, 768), (768, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768), (768, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, 768), (768, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 768), (768, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (3072, 768), (768, 1))
    assert_size_stride(primals_168, (3072, ), (1, ))
    assert_size_stride(primals_169, (768, 3072), (3072, 1))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, 768), (768, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, 768), (768, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, 768), (768, 1))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768), (768, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (3072, 768), (768, 1))
    assert_size_stride(primals_180, (3072, ), (1, ))
    assert_size_stride(primals_181, (768, 3072), (3072, 1))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, 768), (768, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 768), (768, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, 768), (768, 1))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, 768), (768, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (3072, 768), (768, 1))
    assert_size_stride(primals_192, (3072, ), (1, ))
    assert_size_stride(primals_193, (768, 3072), (3072, 1))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_196, (4, 128), (128, 1))
    assert_size_stride(primals_197, (4, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1, 128, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [mask], Original ATen: [aten.unsqueeze]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_unsqueeze_0.run(primals_196, buf0, 65536, grid=grid(65536), stream=stream0)
        buf1 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf6 = reinterpret_tensor(buf4, (4, 128, 1), (128, 1, 1), 0); del buf4  # reuse
        buf7 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf8 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_2, forward, forward_1, l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0, mean, mul, std, sub, truediv, x, x_4], Original ATen: [aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_embedding_mean_mul_std_sub_view_1.run(buf6, primals_196, primals_49, primals_195, primals_197, primals_50, primals_1, primals_2, buf1, buf7, buf8, 512, 768, grid=grid(512), stream=stream0)
        del primals_195
        del primals_2
        del primals_49
        del primals_50
        buf9 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_51, (768, 768), (1, 768), 0), out=buf9)
        buf10 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_53, (768, 768), (1, 768), 0), out=buf10)
        buf11 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), out=buf11)
        buf12 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf9, primals_52, buf12, 393216, grid=grid(393216), stream=stream0)
        del primals_52
        buf13 = reinterpret_tensor(buf9, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf9  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf10, primals_54, buf13, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_54
        buf14 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf13, (48, 64, 128), (8192, 128, 1), 0), out=buf14)
        buf15 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf17 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, eq, p_attn, scores, scores_1], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf14, buf15, buf16, buf17, 6144, 128, grid=grid(6144), stream=stream0)
        buf18 = reinterpret_tensor(buf10, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf10  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf11, primals_56, buf18, 393216, grid=grid(393216), stream=stream0)
        del primals_56
        buf19 = reinterpret_tensor(buf11, (48, 128, 64), (8192, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf17, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf18, (48, 128, 64), (8192, 64, 1), 0), out=buf19)
        buf20 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf19, buf20, 393216, grid=grid(393216), stream=stream0)
        buf21 = reinterpret_tensor(buf19, (512, 768), (768, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_57, (768, 768), (1, 768), 0), out=buf21)
        buf24 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf24, (4, 128, 1), (128, 1, 1), 0); del buf24  # reuse
        buf27 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf28 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, add_6, l__mod___transformer_blocks_0_feed_forward_w_1, mean_2, mul_1, std_2, sub_1, truediv_2, x_7], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf26, buf1, buf21, primals_58, primals_3, primals_4, buf27, buf28, 512, 768, grid=grid(512), stream=stream0)
        del primals_4
        buf29 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, buf28, reinterpret_tensor(primals_59, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf29)
        del primals_60
        buf30 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_activation, l__mod___transformer_blocks_0_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf29, buf30, 1572864, grid=grid(1572864), stream=stream0)
        buf31 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf30, reinterpret_tensor(primals_61, (3072, 768), (1, 3072), 0), out=buf31)
        buf32 = reinterpret_tensor(buf31, (4, 128, 768), (98304, 768, 1), 0); del buf31  # reuse
        buf35 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf37 = reinterpret_tensor(buf35, (4, 128, 1), (128, 1, 1), 0); del buf35  # reuse
        buf38 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf39 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0, mean_4, mul_2, std_4, sub_2, truediv_3, x_12, x_7, x_8], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf32, buf37, buf1, buf21, primals_58, primals_62, primals_5, primals_6, buf38, buf39, 512, 768, grid=grid(512), stream=stream0)
        del primals_58
        del primals_6
        del primals_62
        buf40 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf39, reinterpret_tensor(primals_63, (768, 768), (1, 768), 0), out=buf40)
        buf41 = reinterpret_tensor(buf1, (512, 768), (768, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf39, reinterpret_tensor(primals_65, (768, 768), (1, 768), 0), out=buf41)
        buf42 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf39, reinterpret_tensor(primals_67, (768, 768), (1, 768), 0), out=buf42)
        buf43 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf40, primals_64, buf43, 393216, grid=grid(393216), stream=stream0)
        del primals_64
        buf44 = reinterpret_tensor(buf40, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf40  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf41, primals_66, buf44, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_66
        buf45 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf44, (48, 64, 128), (8192, 128, 1), 0), out=buf45)
        buf46 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf48 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_1, eq, p_attn_2, scores_1, scores_2, scores_3], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf45, buf46, buf47, buf48, 6144, 128, grid=grid(6144), stream=stream0)
        buf49 = reinterpret_tensor(buf41, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf41  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf42, primals_68, buf49, 393216, grid=grid(393216), stream=stream0)
        del primals_68
        buf50 = reinterpret_tensor(buf42, (48, 128, 64), (8192, 64, 1), 0); del buf42  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf49, (48, 128, 64), (8192, 64, 1), 0), out=buf50)
        buf51 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf50, buf51, 393216, grid=grid(393216), stream=stream0)
        buf52 = reinterpret_tensor(buf50, (512, 768), (768, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf51, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), out=buf52)
        buf55 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf57 = reinterpret_tensor(buf55, (4, 128, 1), (128, 1, 1), 0); del buf55  # reuse
        buf58 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf59 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, add_12, l__mod___transformer_blocks_1_feed_forward_w_1, mean_6, mul_3, std_6, sub_3, truediv_5, x_15], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf57, buf32, buf52, primals_70, primals_7, primals_8, buf58, buf59, 512, 768, grid=grid(512), stream=stream0)
        del primals_8
        buf60 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_72, buf59, reinterpret_tensor(primals_71, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf60)
        del primals_72
        buf61 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_activation, l__mod___transformer_blocks_1_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf60, buf61, 1572864, grid=grid(1572864), stream=stream0)
        buf62 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_73, (3072, 768), (1, 3072), 0), out=buf62)
        buf63 = reinterpret_tensor(buf62, (4, 128, 768), (98304, 768, 1), 0); del buf62  # reuse
        buf66 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf68 = reinterpret_tensor(buf66, (4, 128, 1), (128, 1, 1), 0); del buf66  # reuse
        buf69 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf70 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_0, mean_8, mul_4, std_8, sub_4, truediv_6, x_15, x_16, x_20], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf63, buf68, buf32, buf52, primals_70, primals_74, primals_9, primals_10, buf69, buf70, 512, 768, grid=grid(512), stream=stream0)
        del primals_10
        del primals_70
        del primals_74
        buf71 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf70, reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), out=buf71)
        buf72 = reinterpret_tensor(buf32, (512, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf70, reinterpret_tensor(primals_77, (768, 768), (1, 768), 0), out=buf72)
        buf73 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf70, reinterpret_tensor(primals_79, (768, 768), (1, 768), 0), out=buf73)
        buf74 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf71, primals_76, buf74, 393216, grid=grid(393216), stream=stream0)
        del primals_76
        buf75 = reinterpret_tensor(buf71, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf71  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf72, primals_78, buf75, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_78
        buf76 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf75, (48, 64, 128), (8192, 128, 1), 0), out=buf76)
        buf77 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf79 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_2, eq, p_attn_4, scores_1, scores_4, scores_5], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf76, buf77, buf78, buf79, 6144, 128, grid=grid(6144), stream=stream0)
        buf80 = reinterpret_tensor(buf72, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf72  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf73, primals_80, buf80, 393216, grid=grid(393216), stream=stream0)
        del primals_80
        buf81 = reinterpret_tensor(buf73, (48, 128, 64), (8192, 64, 1), 0); del buf73  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf80, (48, 128, 64), (8192, 64, 1), 0), out=buf81)
        buf82 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf81, buf82, 393216, grid=grid(393216), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (512, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(primals_81, (768, 768), (1, 768), 0), out=buf83)
        buf86 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf88 = reinterpret_tensor(buf86, (4, 128, 1), (128, 1, 1), 0); del buf86  # reuse
        buf89 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf90 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, add_18, l__mod___transformer_blocks_2_feed_forward_w_1, mean_10, mul_5, std_10, sub_5, truediv_8, x_23], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf88, buf63, buf83, primals_82, primals_11, primals_12, buf89, buf90, 512, 768, grid=grid(512), stream=stream0)
        del primals_12
        buf91 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_84, buf90, reinterpret_tensor(primals_83, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf91)
        del primals_84
        buf92 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_activation, l__mod___transformer_blocks_2_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf91, buf92, 1572864, grid=grid(1572864), stream=stream0)
        buf93 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf92, reinterpret_tensor(primals_85, (3072, 768), (1, 3072), 0), out=buf93)
        buf94 = reinterpret_tensor(buf93, (4, 128, 768), (98304, 768, 1), 0); del buf93  # reuse
        buf97 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf99 = reinterpret_tensor(buf97, (4, 128, 1), (128, 1, 1), 0); del buf97  # reuse
        buf100 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf101 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_0, mean_12, mul_6, std_12, sub_6, truediv_9, x_23, x_24, x_28], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf94, buf99, buf63, buf83, primals_82, primals_86, primals_13, primals_14, buf100, buf101, 512, 768, grid=grid(512), stream=stream0)
        del primals_14
        del primals_82
        del primals_86
        buf102 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf101, reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), out=buf102)
        buf103 = reinterpret_tensor(buf63, (512, 768), (768, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf101, reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), out=buf103)
        buf104 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf101, reinterpret_tensor(primals_91, (768, 768), (1, 768), 0), out=buf104)
        buf105 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf102, primals_88, buf105, 393216, grid=grid(393216), stream=stream0)
        del primals_88
        buf106 = reinterpret_tensor(buf102, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf102  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf103, primals_90, buf106, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_90
        buf107 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf106, (48, 64, 128), (8192, 128, 1), 0), out=buf107)
        buf108 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf110 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, eq, p_attn_6, scores_1, scores_6, scores_7], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf107, buf108, buf109, buf110, 6144, 128, grid=grid(6144), stream=stream0)
        buf111 = reinterpret_tensor(buf103, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf104, primals_92, buf111, 393216, grid=grid(393216), stream=stream0)
        del primals_92
        buf112 = reinterpret_tensor(buf104, (48, 128, 64), (8192, 64, 1), 0); del buf104  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf111, (48, 128, 64), (8192, 64, 1), 0), out=buf112)
        buf113 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf112, buf113, 393216, grid=grid(393216), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (512, 768), (768, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf113, reinterpret_tensor(primals_93, (768, 768), (1, 768), 0), out=buf114)
        buf117 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf119 = reinterpret_tensor(buf117, (4, 128, 1), (128, 1, 1), 0); del buf117  # reuse
        buf120 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf121 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, add_24, l__mod___transformer_blocks_3_feed_forward_w_1, mean_14, mul_7, std_14, sub_7, truediv_11, x_31], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf119, buf94, buf114, primals_94, primals_15, primals_16, buf120, buf121, 512, 768, grid=grid(512), stream=stream0)
        del primals_16
        buf122 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf121, reinterpret_tensor(primals_95, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf122)
        del primals_96
        buf123 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_activation, l__mod___transformer_blocks_3_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf122, buf123, 1572864, grid=grid(1572864), stream=stream0)
        buf124 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf123, reinterpret_tensor(primals_97, (3072, 768), (1, 3072), 0), out=buf124)
        buf125 = reinterpret_tensor(buf124, (4, 128, 768), (98304, 768, 1), 0); del buf124  # reuse
        buf128 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf130 = reinterpret_tensor(buf128, (4, 128, 1), (128, 1, 1), 0); del buf128  # reuse
        buf131 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf132 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_0, mean_16, mul_8, std_16, sub_8, truediv_12, x_31, x_32, x_36], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf125, buf130, buf94, buf114, primals_94, primals_98, primals_17, primals_18, buf131, buf132, 512, 768, grid=grid(512), stream=stream0)
        del primals_18
        del primals_94
        del primals_98
        buf133 = reinterpret_tensor(buf94, (512, 768), (768, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf132, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), out=buf133)
        buf134 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf132, reinterpret_tensor(primals_101, (768, 768), (1, 768), 0), out=buf134)
        buf135 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf132, reinterpret_tensor(primals_103, (768, 768), (1, 768), 0), out=buf135)
        buf136 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf133, primals_100, buf136, 393216, grid=grid(393216), stream=stream0)
        del primals_100
        buf137 = reinterpret_tensor(buf133, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf133  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf134, primals_102, buf137, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_102
        buf138 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf137, (48, 64, 128), (8192, 128, 1), 0), out=buf138)
        buf139 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf140 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf141 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_4, eq, p_attn_8, scores_1, scores_8, scores_9], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf138, buf139, buf140, buf141, 6144, 128, grid=grid(6144), stream=stream0)
        buf142 = reinterpret_tensor(buf134, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf134  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf135, primals_104, buf142, 393216, grid=grid(393216), stream=stream0)
        del primals_104
        buf143 = reinterpret_tensor(buf135, (48, 128, 64), (8192, 64, 1), 0); del buf135  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf142, (48, 128, 64), (8192, 64, 1), 0), out=buf143)
        buf144 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf143, buf144, 393216, grid=grid(393216), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (512, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf144, reinterpret_tensor(primals_105, (768, 768), (1, 768), 0), out=buf145)
        buf148 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf150 = reinterpret_tensor(buf148, (4, 128, 1), (128, 1, 1), 0); del buf148  # reuse
        buf151 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf152 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, add_30, l__mod___transformer_blocks_4_feed_forward_w_1, mean_18, mul_9, std_18, sub_9, truediv_14, x_39], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf150, buf125, buf145, primals_106, primals_19, primals_20, buf151, buf152, 512, 768, grid=grid(512), stream=stream0)
        del primals_20
        buf153 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_108, buf152, reinterpret_tensor(primals_107, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf153)
        del primals_108
        buf154 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_activation, l__mod___transformer_blocks_4_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf153, buf154, 1572864, grid=grid(1572864), stream=stream0)
        buf155 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf154, reinterpret_tensor(primals_109, (3072, 768), (1, 3072), 0), out=buf155)
        buf156 = reinterpret_tensor(buf155, (4, 128, 768), (98304, 768, 1), 0); del buf155  # reuse
        buf159 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf161 = reinterpret_tensor(buf159, (4, 128, 1), (128, 1, 1), 0); del buf159  # reuse
        buf162 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf163 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_0, mean_20, mul_10, std_20, sub_10, truediv_15, x_39, x_40, x_44], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf156, buf161, buf125, buf145, primals_106, primals_110, primals_21, primals_22, buf162, buf163, 512, 768, grid=grid(512), stream=stream0)
        del primals_106
        del primals_110
        del primals_22
        buf164 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf163, reinterpret_tensor(primals_111, (768, 768), (1, 768), 0), out=buf164)
        buf165 = reinterpret_tensor(buf125, (512, 768), (768, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf163, reinterpret_tensor(primals_113, (768, 768), (1, 768), 0), out=buf165)
        buf166 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf163, reinterpret_tensor(primals_115, (768, 768), (1, 768), 0), out=buf166)
        buf167 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf164, primals_112, buf167, 393216, grid=grid(393216), stream=stream0)
        del primals_112
        buf168 = reinterpret_tensor(buf164, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf164  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf165, primals_114, buf168, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_114
        buf169 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf168, (48, 64, 128), (8192, 128, 1), 0), out=buf169)
        buf170 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf171 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf172 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_5, eq, p_attn_10, scores_1, scores_10, scores_11], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf169, buf170, buf171, buf172, 6144, 128, grid=grid(6144), stream=stream0)
        buf173 = reinterpret_tensor(buf165, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf165  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf166, primals_116, buf173, 393216, grid=grid(393216), stream=stream0)
        del primals_116
        buf174 = reinterpret_tensor(buf166, (48, 128, 64), (8192, 64, 1), 0); del buf166  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf172, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf173, (48, 128, 64), (8192, 64, 1), 0), out=buf174)
        buf175 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf174, buf175, 393216, grid=grid(393216), stream=stream0)
        buf176 = reinterpret_tensor(buf174, (512, 768), (768, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf175, reinterpret_tensor(primals_117, (768, 768), (1, 768), 0), out=buf176)
        buf179 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf181 = reinterpret_tensor(buf179, (4, 128, 1), (128, 1, 1), 0); del buf179  # reuse
        buf182 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf183 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, add_36, l__mod___transformer_blocks_5_feed_forward_w_1, mean_22, mul_11, std_22, sub_11, truediv_17, x_47], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf181, buf156, buf176, primals_118, primals_23, primals_24, buf182, buf183, 512, 768, grid=grid(512), stream=stream0)
        del primals_24
        buf184 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf183, reinterpret_tensor(primals_119, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf184)
        del primals_120
        buf185 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_activation, l__mod___transformer_blocks_5_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf184, buf185, 1572864, grid=grid(1572864), stream=stream0)
        buf186 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf185, reinterpret_tensor(primals_121, (3072, 768), (1, 3072), 0), out=buf186)
        buf187 = reinterpret_tensor(buf186, (4, 128, 768), (98304, 768, 1), 0); del buf186  # reuse
        buf190 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf192 = reinterpret_tensor(buf190, (4, 128, 1), (128, 1, 1), 0); del buf190  # reuse
        buf193 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf194 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_0, mean_24, mul_12, std_24, sub_12, truediv_18, x_47, x_48, x_52], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf187, buf192, buf156, buf176, primals_118, primals_122, primals_25, primals_26, buf193, buf194, 512, 768, grid=grid(512), stream=stream0)
        del primals_118
        del primals_122
        del primals_26
        buf195 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf194, reinterpret_tensor(primals_123, (768, 768), (1, 768), 0), out=buf195)
        buf196 = reinterpret_tensor(buf156, (512, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf194, reinterpret_tensor(primals_125, (768, 768), (1, 768), 0), out=buf196)
        buf197 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf194, reinterpret_tensor(primals_127, (768, 768), (1, 768), 0), out=buf197)
        buf198 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf195, primals_124, buf198, 393216, grid=grid(393216), stream=stream0)
        del primals_124
        buf199 = reinterpret_tensor(buf195, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf195  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf196, primals_126, buf199, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_126
        buf200 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf199, (48, 64, 128), (8192, 128, 1), 0), out=buf200)
        buf201 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf202 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf203 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, eq, p_attn_12, scores_1, scores_12, scores_13], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf200, buf201, buf202, buf203, 6144, 128, grid=grid(6144), stream=stream0)
        buf204 = reinterpret_tensor(buf196, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf196  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf197, primals_128, buf204, 393216, grid=grid(393216), stream=stream0)
        del primals_128
        buf205 = reinterpret_tensor(buf197, (48, 128, 64), (8192, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf203, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf204, (48, 128, 64), (8192, 64, 1), 0), out=buf205)
        buf206 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf205, buf206, 393216, grid=grid(393216), stream=stream0)
        buf207 = reinterpret_tensor(buf205, (512, 768), (768, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf206, reinterpret_tensor(primals_129, (768, 768), (1, 768), 0), out=buf207)
        buf210 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf212 = reinterpret_tensor(buf210, (4, 128, 1), (128, 1, 1), 0); del buf210  # reuse
        buf213 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf214 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_41, add_42, l__mod___transformer_blocks_6_feed_forward_w_1, mean_26, mul_13, std_26, sub_13, truediv_20, x_55], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf212, buf187, buf207, primals_130, primals_27, primals_28, buf213, buf214, 512, 768, grid=grid(512), stream=stream0)
        del primals_28
        buf215 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf214, reinterpret_tensor(primals_131, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf215)
        del primals_132
        buf216 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_activation, l__mod___transformer_blocks_6_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf215, buf216, 1572864, grid=grid(1572864), stream=stream0)
        buf217 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf216, reinterpret_tensor(primals_133, (3072, 768), (1, 3072), 0), out=buf217)
        buf218 = reinterpret_tensor(buf217, (4, 128, 768), (98304, 768, 1), 0); del buf217  # reuse
        buf221 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf223 = reinterpret_tensor(buf221, (4, 128, 1), (128, 1, 1), 0); del buf221  # reuse
        buf224 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf225 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_0, mean_28, mul_14, std_28, sub_14, truediv_21, x_55, x_56, x_60], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf218, buf223, buf187, buf207, primals_130, primals_134, primals_29, primals_30, buf224, buf225, 512, 768, grid=grid(512), stream=stream0)
        del primals_130
        del primals_134
        del primals_30
        buf226 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf225, reinterpret_tensor(primals_135, (768, 768), (1, 768), 0), out=buf226)
        buf227 = reinterpret_tensor(buf187, (512, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf225, reinterpret_tensor(primals_137, (768, 768), (1, 768), 0), out=buf227)
        buf228 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf225, reinterpret_tensor(primals_139, (768, 768), (1, 768), 0), out=buf228)
        buf229 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf226, primals_136, buf229, 393216, grid=grid(393216), stream=stream0)
        del primals_136
        buf230 = reinterpret_tensor(buf226, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf226  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf227, primals_138, buf230, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_138
        buf231 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf230, (48, 64, 128), (8192, 128, 1), 0), out=buf231)
        buf232 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf233 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf234 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_7, eq, p_attn_14, scores_1, scores_14, scores_15], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf231, buf232, buf233, buf234, 6144, 128, grid=grid(6144), stream=stream0)
        buf235 = reinterpret_tensor(buf227, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf227  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf228, primals_140, buf235, 393216, grid=grid(393216), stream=stream0)
        del primals_140
        buf236 = reinterpret_tensor(buf228, (48, 128, 64), (8192, 64, 1), 0); del buf228  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf234, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf235, (48, 128, 64), (8192, 64, 1), 0), out=buf236)
        buf237 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf236, buf237, 393216, grid=grid(393216), stream=stream0)
        buf238 = reinterpret_tensor(buf236, (512, 768), (768, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf237, reinterpret_tensor(primals_141, (768, 768), (1, 768), 0), out=buf238)
        buf241 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf243 = reinterpret_tensor(buf241, (4, 128, 1), (128, 1, 1), 0); del buf241  # reuse
        buf244 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf245 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_47, add_48, l__mod___transformer_blocks_7_feed_forward_w_1, mean_30, mul_15, std_30, sub_15, truediv_23, x_63], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf243, buf218, buf238, primals_142, primals_31, primals_32, buf244, buf245, 512, 768, grid=grid(512), stream=stream0)
        del primals_32
        buf246 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf245, reinterpret_tensor(primals_143, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf246)
        del primals_144
        buf247 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_activation, l__mod___transformer_blocks_7_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf246, buf247, 1572864, grid=grid(1572864), stream=stream0)
        buf248 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf247, reinterpret_tensor(primals_145, (3072, 768), (1, 3072), 0), out=buf248)
        buf249 = reinterpret_tensor(buf248, (4, 128, 768), (98304, 768, 1), 0); del buf248  # reuse
        buf252 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf254 = reinterpret_tensor(buf252, (4, 128, 1), (128, 1, 1), 0); del buf252  # reuse
        buf255 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf256 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_50, l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_0, mean_32, mul_16, std_32, sub_16, truediv_24, x_63, x_64, x_68], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf249, buf254, buf218, buf238, primals_142, primals_146, primals_33, primals_34, buf255, buf256, 512, 768, grid=grid(512), stream=stream0)
        del primals_142
        del primals_146
        del primals_34
        buf257 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf256, reinterpret_tensor(primals_147, (768, 768), (1, 768), 0), out=buf257)
        buf258 = reinterpret_tensor(buf218, (512, 768), (768, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf256, reinterpret_tensor(primals_149, (768, 768), (1, 768), 0), out=buf258)
        buf259 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf256, reinterpret_tensor(primals_151, (768, 768), (1, 768), 0), out=buf259)
        buf260 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf257, primals_148, buf260, 393216, grid=grid(393216), stream=stream0)
        del primals_148
        buf261 = reinterpret_tensor(buf257, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf257  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf258, primals_150, buf261, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_150
        buf262 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf261, (48, 64, 128), (8192, 128, 1), 0), out=buf262)
        buf263 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf265 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_8, eq, p_attn_16, scores_1, scores_16, scores_17], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf262, buf263, buf264, buf265, 6144, 128, grid=grid(6144), stream=stream0)
        buf266 = reinterpret_tensor(buf258, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf259, primals_152, buf266, 393216, grid=grid(393216), stream=stream0)
        del primals_152
        buf267 = reinterpret_tensor(buf259, (48, 128, 64), (8192, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf266, (48, 128, 64), (8192, 64, 1), 0), out=buf267)
        buf268 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf267, buf268, 393216, grid=grid(393216), stream=stream0)
        buf269 = reinterpret_tensor(buf267, (512, 768), (768, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf268, reinterpret_tensor(primals_153, (768, 768), (1, 768), 0), out=buf269)
        buf272 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf274 = reinterpret_tensor(buf272, (4, 128, 1), (128, 1, 1), 0); del buf272  # reuse
        buf275 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf276 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53, add_54, l__mod___transformer_blocks_8_feed_forward_w_1, mean_34, mul_17, std_34, sub_17, truediv_26, x_71], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf274, buf249, buf269, primals_154, primals_35, primals_36, buf275, buf276, 512, 768, grid=grid(512), stream=stream0)
        del primals_36
        buf277 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_156, buf276, reinterpret_tensor(primals_155, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf277)
        del primals_156
        buf278 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_activation, l__mod___transformer_blocks_8_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf277, buf278, 1572864, grid=grid(1572864), stream=stream0)
        buf279 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf278, reinterpret_tensor(primals_157, (3072, 768), (1, 3072), 0), out=buf279)
        buf280 = reinterpret_tensor(buf279, (4, 128, 768), (98304, 768, 1), 0); del buf279  # reuse
        buf283 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf285 = reinterpret_tensor(buf283, (4, 128, 1), (128, 1, 1), 0); del buf283  # reuse
        buf286 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf287 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_56, l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_0, mean_36, mul_18, std_36, sub_18, truediv_27, x_71, x_72, x_76], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf280, buf285, buf249, buf269, primals_154, primals_158, primals_37, primals_38, buf286, buf287, 512, 768, grid=grid(512), stream=stream0)
        del primals_154
        del primals_158
        del primals_38
        buf288 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf287, reinterpret_tensor(primals_159, (768, 768), (1, 768), 0), out=buf288)
        buf289 = reinterpret_tensor(buf249, (512, 768), (768, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf287, reinterpret_tensor(primals_161, (768, 768), (1, 768), 0), out=buf289)
        buf290 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf287, reinterpret_tensor(primals_163, (768, 768), (1, 768), 0), out=buf290)
        buf291 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf288, primals_160, buf291, 393216, grid=grid(393216), stream=stream0)
        del primals_160
        buf292 = reinterpret_tensor(buf288, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf288  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf289, primals_162, buf292, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_162
        buf293 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf292, (48, 64, 128), (8192, 128, 1), 0), out=buf293)
        buf294 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf295 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cuda', dtype=torch.float32)
        buf296 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_9, eq, p_attn_18, scores_1, scores_18, scores_19], Original ATen: [aten._softmax, aten.clone, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_div_eq_masked_fill_4.run(buf0, buf293, buf294, buf295, buf296, 6144, 128, grid=grid(6144), stream=stream0)
        buf297 = reinterpret_tensor(buf289, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf289  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf290, primals_164, buf297, 393216, grid=grid(393216), stream=stream0)
        del primals_164
        buf298 = reinterpret_tensor(buf290, (48, 128, 64), (8192, 64, 1), 0); del buf290  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf297, (48, 128, 64), (8192, 64, 1), 0), out=buf298)
        buf299 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf298, buf299, 393216, grid=grid(393216), stream=stream0)
        buf300 = reinterpret_tensor(buf298, (512, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf299, reinterpret_tensor(primals_165, (768, 768), (1, 768), 0), out=buf300)
        buf303 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf305 = reinterpret_tensor(buf303, (4, 128, 1), (128, 1, 1), 0); del buf303  # reuse
        buf306 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf307 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_59, add_60, l__mod___transformer_blocks_9_feed_forward_w_1, mean_38, mul_19, std_38, sub_19, truediv_29, x_79], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf305, buf280, buf300, primals_166, primals_39, primals_40, buf306, buf307, 512, 768, grid=grid(512), stream=stream0)
        del primals_40
        buf308 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_168, buf307, reinterpret_tensor(primals_167, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf308)
        del primals_168
        buf309 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_activation, l__mod___transformer_blocks_9_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf308, buf309, 1572864, grid=grid(1572864), stream=stream0)
        buf310 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf309, reinterpret_tensor(primals_169, (3072, 768), (1, 3072), 0), out=buf310)
        buf311 = reinterpret_tensor(buf310, (4, 128, 768), (98304, 768, 1), 0); del buf310  # reuse
        buf314 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf316 = reinterpret_tensor(buf314, (4, 128, 1), (128, 1, 1), 0); del buf314  # reuse
        buf317 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf318 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_0, mean_40, mul_20, std_40, sub_20, truediv_30, x_79, x_80, x_84], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf311, buf316, buf280, buf300, primals_166, primals_170, primals_41, primals_42, buf317, buf318, 512, 768, grid=grid(512), stream=stream0)
        del primals_166
        del primals_170
        del primals_42
        buf319 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf318, reinterpret_tensor(primals_171, (768, 768), (1, 768), 0), out=buf319)
        buf320 = reinterpret_tensor(buf280, (512, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf318, reinterpret_tensor(primals_173, (768, 768), (1, 768), 0), out=buf320)
        buf321 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf318, reinterpret_tensor(primals_175, (768, 768), (1, 768), 0), out=buf321)
        buf322 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf319, primals_172, buf322, 393216, grid=grid(393216), stream=stream0)
        del primals_172
        buf323 = reinterpret_tensor(buf319, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf319  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf320, primals_174, buf323, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_174
        buf324 = empty((48, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf322, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf323, (48, 64, 128), (8192, 128, 1), 0), out=buf324)
        buf327 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        buf375 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        buf376 = reinterpret_tensor(buf293, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf293  # reuse
        buf377 = reinterpret_tensor(buf262, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf262  # reuse
        buf378 = reinterpret_tensor(buf231, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf231  # reuse
        buf379 = reinterpret_tensor(buf200, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf200  # reuse
        buf380 = reinterpret_tensor(buf169, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf169  # reuse
        buf381 = reinterpret_tensor(buf138, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf138  # reuse
        buf382 = reinterpret_tensor(buf107, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf107  # reuse
        buf383 = reinterpret_tensor(buf76, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf76  # reuse
        buf384 = reinterpret_tensor(buf45, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf45  # reuse
        buf385 = reinterpret_tensor(buf14, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf14  # reuse
        # Source Nodes: [attn_10, eq, p_attn, p_attn_10, p_attn_12, p_attn_14, p_attn_16, p_attn_18, p_attn_2, p_attn_20, p_attn_4, p_attn_6, p_attn_8, scores, scores_1, scores_10, scores_11, scores_12, scores_13, scores_14, scores_15, scores_16, scores_17, scores_18, scores_19, scores_2, scores_20, scores_21, scores_3, scores_4, scores_5, scores_6, scores_7, scores_8, scores_9], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_detach_div_eq_masked_fill_9.run(buf376, buf377, buf378, buf379, buf380, buf381, buf382, buf383, buf384, buf385, buf0, buf324, buf294, buf295, buf263, buf264, buf232, buf233, buf201, buf202, buf170, buf171, buf139, buf140, buf108, buf109, buf77, buf78, buf46, buf47, buf15, buf16, buf327, buf375, 6144, 128, grid=grid(6144), stream=stream0)
        del buf108
        del buf109
        del buf139
        del buf140
        del buf15
        del buf16
        del buf170
        del buf171
        del buf201
        del buf202
        del buf232
        del buf233
        del buf263
        del buf264
        del buf294
        del buf295
        del buf46
        del buf47
        del buf77
        del buf78
        buf328 = reinterpret_tensor(buf320, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf320  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf321, primals_176, buf328, 393216, grid=grid(393216), stream=stream0)
        del primals_176
        buf329 = reinterpret_tensor(buf321, (48, 128, 64), (8192, 64, 1), 0); del buf321  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf327, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf328, (48, 128, 64), (8192, 64, 1), 0), out=buf329)
        buf330 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf329, buf330, 393216, grid=grid(393216), stream=stream0)
        buf331 = reinterpret_tensor(buf329, (512, 768), (768, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf330, reinterpret_tensor(primals_177, (768, 768), (1, 768), 0), out=buf331)
        buf334 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf336 = reinterpret_tensor(buf334, (4, 128, 1), (128, 1, 1), 0); del buf334  # reuse
        buf337 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf338 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_65, add_66, l__mod___transformer_blocks_10_feed_forward_w_1, mean_42, mul_21, std_42, sub_21, truediv_32, x_87], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf336, buf311, buf331, primals_178, primals_43, primals_44, buf337, buf338, 512, 768, grid=grid(512), stream=stream0)
        del primals_44
        buf339 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_180, buf338, reinterpret_tensor(primals_179, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf339)
        del primals_180
        buf340 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_activation, l__mod___transformer_blocks_10_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf339, buf340, 1572864, grid=grid(1572864), stream=stream0)
        buf341 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf340, reinterpret_tensor(primals_181, (3072, 768), (1, 3072), 0), out=buf341)
        buf342 = reinterpret_tensor(buf341, (4, 128, 768), (98304, 768, 1), 0); del buf341  # reuse
        buf345 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf347 = reinterpret_tensor(buf345, (4, 128, 1), (128, 1, 1), 0); del buf345  # reuse
        buf348 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf349 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_0, mean_44, mul_22, std_44, sub_22, truediv_33, x_87, x_88, x_92], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_8.run(buf342, buf347, buf311, buf331, primals_178, primals_182, primals_45, primals_46, buf348, buf349, 512, 768, grid=grid(512), stream=stream0)
        del primals_178
        del primals_182
        del primals_46
        buf350 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf349, reinterpret_tensor(primals_183, (768, 768), (1, 768), 0), out=buf350)
        buf351 = reinterpret_tensor(buf311, (512, 768), (768, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf349, reinterpret_tensor(primals_185, (768, 768), (1, 768), 0), out=buf351)
        buf352 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf349, reinterpret_tensor(primals_187, (768, 768), (1, 768), 0), out=buf352)
        buf353 = empty((4, 12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf350, primals_184, buf353, 393216, grid=grid(393216), stream=stream0)
        del primals_184
        buf354 = reinterpret_tensor(buf350, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf350  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf351, primals_186, buf354, 3072, 128, grid=grid(3072, 128), stream=stream0)
        del primals_186
        buf355 = buf324; del buf324  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf353, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf354, (48, 64, 128), (8192, 128, 1), 0), out=buf355)
        buf358 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        buf374 = empty((4, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_11, eq, p_attn_22, scores_1, scores_22, scores_23], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.div, aten.eq, aten.masked_fill]
        triton_per_fused__softmax_clone_detach_div_eq_masked_fill_10.run(buf0, buf355, buf358, buf374, 6144, 128, grid=grid(6144), stream=stream0)
        del buf355
        buf359 = reinterpret_tensor(buf351, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf351  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf352, primals_188, buf359, 393216, grid=grid(393216), stream=stream0)
        del primals_188
        buf360 = reinterpret_tensor(buf352, (48, 128, 64), (8192, 64, 1), 0); del buf352  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf358, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf359, (48, 128, 64), (8192, 64, 1), 0), out=buf360)
        buf361 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_output_linear], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf360, buf361, 393216, grid=grid(393216), stream=stream0)
        buf362 = reinterpret_tensor(buf360, (512, 768), (768, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf361, reinterpret_tensor(primals_189, (768, 768), (1, 768), 0), out=buf362)
        buf365 = empty_strided((4, 128, 1), (128, 1, 512), device='cuda', dtype=torch.float32)
        buf367 = reinterpret_tensor(buf365, (4, 128, 1), (128, 1, 1), 0); del buf365  # reuse
        buf368 = empty((4, 128, 768), device='cuda', dtype=torch.float32)
        buf369 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_71, add_72, l__mod___transformer_blocks_11_feed_forward_w_1, mean_46, mul_23, std_46, sub_23, truediv_35, x_95], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.std, aten.sub, aten.view]
        triton_per_fused_add_div_mean_mul_std_sub_view_6.run(buf367, buf342, buf362, primals_190, primals_47, primals_48, buf368, buf369, 512, 768, grid=grid(512), stream=stream0)
        del primals_48
        buf370 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_w_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_192, buf369, reinterpret_tensor(primals_191, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf370)
        del primals_192
        buf371 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_activation, l__mod___transformer_blocks_11_feed_forward_w_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf370, buf371, 1572864, grid=grid(1572864), stream=stream0)
        buf372 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf371, reinterpret_tensor(primals_193, (3072, 768), (1, 3072), 0), out=buf372)
        buf373 = reinterpret_tensor(buf372, (4, 128, 768), (98304, 768, 1), 0); del buf372  # reuse
        # Source Nodes: [x_95, x_96], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf373, buf342, buf362, primals_190, primals_194, 393216, grid=grid(393216), stream=stream0)
        del buf342
        del buf362
        del primals_190
        del primals_194
        return (buf373, buf0, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_196, primals_197, buf0, buf6, buf7, buf8, buf20, buf26, buf27, buf28, buf29, buf30, buf37, buf38, buf39, buf51, buf57, buf58, buf59, buf60, buf61, buf68, buf69, buf70, buf82, buf88, buf89, buf90, buf91, buf92, buf99, buf100, buf101, buf113, buf119, buf120, buf121, buf122, buf123, buf130, buf131, buf132, buf144, buf150, buf151, buf152, buf153, buf154, buf161, buf162, buf163, buf175, buf181, buf182, buf183, buf184, buf185, buf192, buf193, buf194, buf206, buf212, buf213, buf214, buf215, buf216, buf223, buf224, buf225, buf237, buf243, buf244, buf245, buf246, buf247, buf254, buf255, buf256, buf268, buf274, buf275, buf276, buf277, buf278, buf285, buf286, buf287, buf299, buf305, buf306, buf307, buf308, buf309, buf316, buf317, buf318, buf330, buf336, buf337, buf338, buf339, buf340, buf347, buf348, buf349, buf361, buf367, buf368, buf369, buf370, buf371, reinterpret_tensor(primals_193, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_191, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_189, (768, 768), (768, 1), 0), reinterpret_tensor(buf358, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf359, (48, 64, 128), (8192, 1, 64), 0), buf374, reinterpret_tensor(buf353, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf354, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_187, (768, 768), (768, 1), 0), reinterpret_tensor(primals_185, (768, 768), (768, 1), 0), reinterpret_tensor(primals_183, (768, 768), (768, 1), 0), reinterpret_tensor(primals_181, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_179, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_177, (768, 768), (768, 1), 0), reinterpret_tensor(buf327, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf328, (48, 64, 128), (8192, 1, 64), 0), buf375, reinterpret_tensor(buf322, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf323, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_175, (768, 768), (768, 1), 0), reinterpret_tensor(primals_173, (768, 768), (768, 1), 0), reinterpret_tensor(primals_171, (768, 768), (768, 1), 0), reinterpret_tensor(primals_169, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_167, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_165, (768, 768), (768, 1), 0), reinterpret_tensor(buf296, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf297, (48, 64, 128), (8192, 1, 64), 0), buf376, reinterpret_tensor(buf291, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf292, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_163, (768, 768), (768, 1), 0), reinterpret_tensor(primals_161, (768, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 768), (768, 1), 0), reinterpret_tensor(primals_157, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_155, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_153, (768, 768), (768, 1), 0), reinterpret_tensor(buf265, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf266, (48, 64, 128), (8192, 1, 64), 0), buf377, reinterpret_tensor(buf260, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf261, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_151, (768, 768), (768, 1), 0), reinterpret_tensor(primals_149, (768, 768), (768, 1), 0), reinterpret_tensor(primals_147, (768, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_141, (768, 768), (768, 1), 0), reinterpret_tensor(buf234, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf235, (48, 64, 128), (8192, 1, 64), 0), buf378, reinterpret_tensor(buf229, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf230, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_139, (768, 768), (768, 1), 0), reinterpret_tensor(primals_137, (768, 768), (768, 1), 0), reinterpret_tensor(primals_135, (768, 768), (768, 1), 0), reinterpret_tensor(primals_133, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_129, (768, 768), (768, 1), 0), reinterpret_tensor(buf203, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf204, (48, 64, 128), (8192, 1, 64), 0), buf379, reinterpret_tensor(buf198, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf199, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_127, (768, 768), (768, 1), 0), reinterpret_tensor(primals_125, (768, 768), (768, 1), 0), reinterpret_tensor(primals_123, (768, 768), (768, 1), 0), reinterpret_tensor(primals_121, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_119, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_117, (768, 768), (768, 1), 0), reinterpret_tensor(buf172, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf173, (48, 64, 128), (8192, 1, 64), 0), buf380, reinterpret_tensor(buf167, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf168, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_115, (768, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 768), (768, 1), 0), reinterpret_tensor(primals_111, (768, 768), (768, 1), 0), reinterpret_tensor(primals_109, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_105, (768, 768), (768, 1), 0), reinterpret_tensor(buf141, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf142, (48, 64, 128), (8192, 1, 64), 0), buf381, reinterpret_tensor(buf136, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf137, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_103, (768, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 768), (768, 1), 0), reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_93, (768, 768), (768, 1), 0), reinterpret_tensor(buf110, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf111, (48, 64, 128), (8192, 1, 64), 0), buf382, reinterpret_tensor(buf105, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf106, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_91, (768, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 768), (768, 1), 0), reinterpret_tensor(primals_87, (768, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 768), (768, 1), 0), reinterpret_tensor(buf79, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf80, (48, 64, 128), (8192, 1, 64), 0), buf383, reinterpret_tensor(buf74, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf75, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_79, (768, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 768), (768, 1), 0), reinterpret_tensor(primals_75, (768, 768), (768, 1), 0), reinterpret_tensor(primals_73, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), reinterpret_tensor(buf48, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf49, (48, 64, 128), (8192, 1, 64), 0), buf384, reinterpret_tensor(buf43, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf44, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_67, (768, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 768), (768, 1), 0), reinterpret_tensor(primals_63, (768, 768), (768, 1), 0), reinterpret_tensor(primals_61, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_57, (768, 768), (768, 1), 0), reinterpret_tensor(buf17, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf18, (48, 64, 128), (8192, 1, 64), 0), buf385, reinterpret_tensor(buf12, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf13, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_55, (768, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 768), (768, 1), 0), reinterpret_tensor(primals_51, (768, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((20005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((3, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_197 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BERT_pytorch', benchmark_compiled_module)
