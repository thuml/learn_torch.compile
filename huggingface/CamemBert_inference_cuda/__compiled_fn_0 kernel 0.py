
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


# kernel path: /tmp/torchinductor_youkaichao/k7/ck77kdncnxybcbyhmk7ki3jxwwrrpqne3wghsuhv4yaccphu7lpw.py
# Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
# cumsum => cumsum
# mask => convert_element_type
# ne => ne
triton_poi_fused__to_copy_cumsum_ne_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gzyuvvbpajo6okxqkmgx4ejhcjzh5pyki67kgznsf6jxe5p2l3.py
# Source Nodes: [add, embeddings, embeddings_1, embeddings_2, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, position_ids, token_type_embeddings, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.ne]
# add => add
# embeddings => add_2
# embeddings_1 => add_3
# embeddings_2 => add_4, add_5, mul_2, mul_3, rsqrt, sub_1, var_mean
# incremental_indices => mul_1
# inputs_embeds => embedding
# long => convert_element_type_2
# mask => convert_element_type
# ne => ne
# position_embeddings => embedding_2
# position_ids => add_1
# token_type_embeddings => embedding_1
# type_as => convert_element_type_1
triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32005
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 32005)) | ~xmask, "index out of bounds: 0 <= tmp3 < 32005")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 1)) | ~xmask, "index out of bounds: 0 <= tmp8 < 1")
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11.to(tl.int32)
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp0 != tmp15
    tmp17 = tmp16.to(tl.int32)
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18.to(tl.int64)
    tmp20 = tmp19 + tmp15
    tmp21 = tmp20 + 514
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tl.device_assert(((0 <= tmp23) & (tmp23 < 514)) | ~xmask, "index out of bounds: 0 <= tmp23 < 514")
    tmp24 = tl.load(in_ptr5 + (r1 + (768*tmp23)), rmask & xmask, other=0.0)
    tmp25 = tmp10 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tl.full([1], 768, tl.int32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 / tmp34
    tmp36 = tmp26 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp42 = tmp25 - tmp35
    tmp43 = 768.0
    tmp44 = tmp41 / tmp43
    tmp45 = 1e-05
    tmp46 = tmp44 + tmp45
    tmp47 = tl.math.rsqrt(tmp46)
    tmp48 = tmp42 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp52, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztmswonzytrs4gr6tiyjolh437pxweu4zijl5gf5flkr5jtxn3g.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3e2embrdyayjzc3mkhw2m2dmzhwrax723uc6pdx5o7ohemn654l.py
# Source Nodes: [add_4, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_4 => add_7
# attention_output => add_8, add_9, mul_4, mul_5, rsqrt_1, sub_3, var_mean_1
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': []}
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/csk2amice572reqodh3jpnnjrppxs57nr22y2qljewdne2u2jf4f.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_10, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4oi2zimra5hicxhrfi7kvzea4isedm6chbcflhwttyro5mr523.py
# Source Nodes: [x_37, x_38], Original ATen: [aten.gelu, aten.native_layer_norm]
# x_37 => add_102, erf_12, mul_88, mul_89, mul_90
# x_38 => add_103, add_104, mul_91, mul_92, rsqrt_25, sub_38, var_mean_25
triton_per_fused_gelu_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36xit7fkpsvr27jfki7arcekh2g3qdizs3cj7wehdzlfklio72v.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_12, exp_12, sub_39, sum_13
triton_red_fused__log_softmax_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32005
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32005*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (32005*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctw3gac3542gukbhycgpmacnixdrryzu5vyxpdmjwmqabctoiizm.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type_3, div_24, full_default_2, ne_2, ne_3, neg, sum_14, sum_15, where_1
triton_per_fused_nll_loss_forward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 32005
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 32005), "index out of bounds: 0 <= tmp7 < 32005")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (32005*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.log(tmp11)
    tmp13 = tmp10 - tmp12
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp2.to(tl.int64)
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp27, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32005, 768), (768, 1))
    assert_size_stride(arg1_1, (1, 768), (768, 1))
    assert_size_stride(arg2_1, (514, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (3072, ), (1, ))
    assert_size_stride(arg17_1, (768, 3072), (3072, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (3072, ), (1, ))
    assert_size_stride(arg33_1, (768, 3072), (3072, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, 768), (768, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, 768), (768, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (3072, ), (1, ))
    assert_size_stride(arg49_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, 768), (768, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 768), (768, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, 768), (768, 1))
    assert_size_stride(arg64_1, (3072, ), (1, ))
    assert_size_stride(arg65_1, (768, 3072), (3072, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (3072, ), (1, ))
    assert_size_stride(arg81_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, 768), (768, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (3072, ), (1, ))
    assert_size_stride(arg97_1, (768, 3072), (3072, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, 768), (768, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, 768), (768, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (3072, 768), (768, 1))
    assert_size_stride(arg112_1, (3072, ), (1, ))
    assert_size_stride(arg113_1, (768, 3072), (3072, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, 768), (768, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, 768), (768, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, 768), (768, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (3072, 768), (768, 1))
    assert_size_stride(arg128_1, (3072, ), (1, ))
    assert_size_stride(arg129_1, (768, 3072), (3072, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, 768), (768, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, 768), (768, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (3072, 768), (768, 1))
    assert_size_stride(arg144_1, (3072, ), (1, ))
    assert_size_stride(arg145_1, (768, 3072), (3072, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, 768), (768, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, 768), (768, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, 768), (768, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (3072, 768), (768, 1))
    assert_size_stride(arg160_1, (3072, ), (1, ))
    assert_size_stride(arg161_1, (768, 3072), (3072, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, 768), (768, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, 768), (768, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, 768), (768, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, 768), (768, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (3072, 768), (768, 1))
    assert_size_stride(arg176_1, (3072, ), (1, ))
    assert_size_stride(arg177_1, (768, 3072), (3072, 1))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, 768), (768, 1))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, 768), (768, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, 768), (768, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (3072, 768), (768, 1))
    assert_size_stride(arg192_1, (3072, ), (1, ))
    assert_size_stride(arg193_1, (768, 3072), (3072, 1))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, 768), (768, 1))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (32005, 768), (768, 1))
    assert_size_stride(arg202_1, (32005, ), (1, ))
    assert_size_stride(arg203_1, (1, 514), (514, 1))
    assert_size_stride(arg204_1, (1, 512), (512, 1))
    assert_size_stride(arg205_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.int32)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_cumsum_ne_0.run(arg204_1, buf0, 512, grid=grid(512), stream=stream0)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf1 = aten.cumsum(buf0, 1)
        del buf0
        buf2 = buf1
        del buf1
        buf3 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf7 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embeddings, embeddings_1, embeddings_2, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, position_ids, token_type_embeddings, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.ne]
        triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_1.run(arg204_1, arg0_1, arg203_1, arg1_1, buf2, arg2_1, arg3_1, arg4_1, buf3, buf7, 512, 768, grid=grid(512), stream=stream0)
        del arg0_1
        del arg1_1
        del arg203_1
        del arg204_1
        del arg2_1
        del arg3_1
        del arg4_1
        del buf2
        buf8 = reinterpret_tensor(buf3, (512, 768), (768, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), out=buf8)
        del arg5_1
        buf9 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), out=buf9)
        del arg7_1
        buf10 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf10)
        del arg9_1
        buf11 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf8, arg6_1, buf11, 393216, grid=grid(393216), stream=stream0)
        del arg6_1
        buf12 = reinterpret_tensor(buf8, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf9, arg8_1, buf12, 393216, grid=grid(393216), stream=stream0)
        del arg8_1
        buf13 = reinterpret_tensor(buf9, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf10, arg10_1, buf13, 393216, grid=grid(393216), stream=stream0)
        del arg10_1
        del buf10
        # Source Nodes: [], Original ATen: []
        buf14 = aten._scaled_dot_product_efficient_attention(buf11, buf12, buf13, None, False, scale=0.125)
        buf15 = buf14[0]
        del buf14
        buf19 = reinterpret_tensor(buf13, (512, 768), (768, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (512, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), out=buf19)
        del arg11_1
        buf23 = reinterpret_tensor(buf15, (1, 512, 768), (393216, 768, 1), 0); del buf15  # reuse
        # Source Nodes: [add_4, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf19, arg12_1, buf7, arg13_1, arg14_1, buf23, 512, 768, grid=grid(512), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        buf24 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 3072), (1, 768), 0), out=buf24)
        del arg15_1
        buf25 = reinterpret_tensor(buf24, (1, 512, 3072), (1572864, 3072, 1), 0); del buf24  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf25, arg16_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg16_1
        buf26 = reinterpret_tensor(buf7, (512, 768), (768, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg17_1, (3072, 768), (1, 3072), 0), out=buf26)
        del arg17_1
        buf30 = reinterpret_tensor(buf19, (1, 512, 768), (393216, 768, 1), 0); del buf19  # reuse
        # Source Nodes: [add_5, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf26, arg18_1, buf23, arg19_1, arg20_1, buf30, 512, 768, grid=grid(512), stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        buf31 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), out=buf31)
        del arg21_1
        buf32 = reinterpret_tensor(buf23, (512, 768), (768, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), out=buf32)
        del arg23_1
        buf33 = reinterpret_tensor(buf12, (512, 768), (768, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), out=buf33)
        del arg25_1
        buf34 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf31, arg22_1, buf34, 393216, grid=grid(393216), stream=stream0)
        del arg22_1
        buf35 = reinterpret_tensor(buf31, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf32, arg24_1, buf35, 393216, grid=grid(393216), stream=stream0)
        del arg24_1
        buf36 = reinterpret_tensor(buf32, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf33, arg26_1, buf36, 393216, grid=grid(393216), stream=stream0)
        del arg26_1
        del buf33
        # Source Nodes: [], Original ATen: []
        buf37 = aten._scaled_dot_product_efficient_attention(buf34, buf35, buf36, None, False, scale=0.125)
        buf38 = buf37[0]
        del buf37
        buf42 = reinterpret_tensor(buf36, (512, 768), (768, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (512, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), out=buf42)
        del arg27_1
        buf46 = reinterpret_tensor(buf38, (1, 512, 768), (393216, 768, 1), 0); del buf38  # reuse
        # Source Nodes: [add_7, attention_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf42, arg28_1, buf30, arg29_1, arg30_1, buf46, 512, 768, grid=grid(512), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        buf47 = reinterpret_tensor(buf25, (512, 3072), (3072, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (512, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 3072), (1, 768), 0), out=buf47)
        del arg31_1
        buf48 = reinterpret_tensor(buf47, (1, 512, 3072), (1572864, 3072, 1), 0); del buf47  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf48, arg32_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg32_1
        buf49 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg33_1, (3072, 768), (1, 3072), 0), out=buf49)
        del arg33_1
        buf53 = buf30; del buf30  # reuse
        # Source Nodes: [add_8, hidden_states_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf49, arg34_1, buf46, arg35_1, arg36_1, buf53, 512, 768, grid=grid(512), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf54 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 768), (1, 768), 0), out=buf54)
        del arg37_1
        buf55 = reinterpret_tensor(buf46, (512, 768), (768, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), out=buf55)
        del arg39_1
        buf56 = reinterpret_tensor(buf35, (512, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), out=buf56)
        del arg41_1
        buf57 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf54, arg38_1, buf57, 393216, grid=grid(393216), stream=stream0)
        del arg38_1
        buf58 = reinterpret_tensor(buf54, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf55, arg40_1, buf58, 393216, grid=grid(393216), stream=stream0)
        del arg40_1
        buf59 = reinterpret_tensor(buf55, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf56, arg42_1, buf59, 393216, grid=grid(393216), stream=stream0)
        del arg42_1
        del buf56
        # Source Nodes: [], Original ATen: []
        buf60 = aten._scaled_dot_product_efficient_attention(buf57, buf58, buf59, None, False, scale=0.125)
        buf61 = buf60[0]
        del buf60
        buf65 = reinterpret_tensor(buf59, (512, 768), (768, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (512, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), out=buf65)
        del arg43_1
        buf69 = reinterpret_tensor(buf61, (1, 512, 768), (393216, 768, 1), 0); del buf61  # reuse
        # Source Nodes: [add_10, attention_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf65, arg44_1, buf53, arg45_1, arg46_1, buf69, 512, 768, grid=grid(512), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf70 = reinterpret_tensor(buf48, (512, 3072), (3072, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), out=buf70)
        del arg47_1
        buf71 = reinterpret_tensor(buf70, (1, 512, 3072), (1572864, 3072, 1), 0); del buf70  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf71, arg48_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg48_1
        buf72 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), out=buf72)
        del arg49_1
        buf76 = buf53; del buf53  # reuse
        # Source Nodes: [add_11, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf72, arg50_1, buf69, arg51_1, arg52_1, buf76, 512, 768, grid=grid(512), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf77 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 768), (1, 768), 0), out=buf77)
        del arg53_1
        buf78 = reinterpret_tensor(buf69, (512, 768), (768, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), out=buf78)
        del arg55_1
        buf79 = reinterpret_tensor(buf58, (512, 768), (768, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf79)
        del arg57_1
        buf80 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, arg54_1, buf80, 393216, grid=grid(393216), stream=stream0)
        del arg54_1
        buf81 = reinterpret_tensor(buf77, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, arg56_1, buf81, 393216, grid=grid(393216), stream=stream0)
        del arg56_1
        buf82 = reinterpret_tensor(buf78, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf79, arg58_1, buf82, 393216, grid=grid(393216), stream=stream0)
        del arg58_1
        del buf79
        # Source Nodes: [], Original ATen: []
        buf83 = aten._scaled_dot_product_efficient_attention(buf80, buf81, buf82, None, False, scale=0.125)
        buf84 = buf83[0]
        del buf83
        buf88 = reinterpret_tensor(buf82, (512, 768), (768, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (512, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), out=buf88)
        del arg59_1
        buf92 = reinterpret_tensor(buf84, (1, 512, 768), (393216, 768, 1), 0); del buf84  # reuse
        # Source Nodes: [add_13, attention_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf88, arg60_1, buf76, arg61_1, arg62_1, buf92, 512, 768, grid=grid(512), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        buf93 = reinterpret_tensor(buf71, (512, 3072), (3072, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (512, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 3072), (1, 768), 0), out=buf93)
        del arg63_1
        buf94 = reinterpret_tensor(buf93, (1, 512, 3072), (1572864, 3072, 1), 0); del buf93  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf94, arg64_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg64_1
        buf95 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg65_1, (3072, 768), (1, 3072), 0), out=buf95)
        del arg65_1
        buf99 = buf76; del buf76  # reuse
        # Source Nodes: [add_14, hidden_states_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf95, arg66_1, buf92, arg67_1, arg68_1, buf99, 512, 768, grid=grid(512), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        buf100 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), out=buf100)
        del arg69_1
        buf101 = reinterpret_tensor(buf92, (512, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), out=buf101)
        del arg71_1
        buf102 = reinterpret_tensor(buf81, (512, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), out=buf102)
        del arg73_1
        buf103 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf100, arg70_1, buf103, 393216, grid=grid(393216), stream=stream0)
        del arg70_1
        buf104 = reinterpret_tensor(buf100, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf101, arg72_1, buf104, 393216, grid=grid(393216), stream=stream0)
        del arg72_1
        buf105 = reinterpret_tensor(buf101, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf102, arg74_1, buf105, 393216, grid=grid(393216), stream=stream0)
        del arg74_1
        del buf102
        # Source Nodes: [], Original ATen: []
        buf106 = aten._scaled_dot_product_efficient_attention(buf103, buf104, buf105, None, False, scale=0.125)
        buf107 = buf106[0]
        del buf106
        buf111 = reinterpret_tensor(buf105, (512, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (512, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), out=buf111)
        del arg75_1
        buf115 = reinterpret_tensor(buf107, (1, 512, 768), (393216, 768, 1), 0); del buf107  # reuse
        # Source Nodes: [add_16, attention_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf111, arg76_1, buf99, arg77_1, arg78_1, buf115, 512, 768, grid=grid(512), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf116 = reinterpret_tensor(buf94, (512, 3072), (3072, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), out=buf116)
        del arg79_1
        buf117 = reinterpret_tensor(buf116, (1, 512, 3072), (1572864, 3072, 1), 0); del buf116  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf117, arg80_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg80_1
        buf118 = reinterpret_tensor(buf99, (512, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), out=buf118)
        del arg81_1
        buf122 = reinterpret_tensor(buf111, (1, 512, 768), (393216, 768, 1), 0); del buf111  # reuse
        # Source Nodes: [add_17, hidden_states_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf118, arg82_1, buf115, arg83_1, arg84_1, buf122, 512, 768, grid=grid(512), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        buf123 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (512, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), out=buf123)
        del arg85_1
        buf124 = reinterpret_tensor(buf115, (512, 768), (768, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (512, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf124)
        del arg87_1
        buf125 = reinterpret_tensor(buf104, (512, 768), (768, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (512, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), out=buf125)
        del arg89_1
        buf126 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf123, arg86_1, buf126, 393216, grid=grid(393216), stream=stream0)
        del arg86_1
        buf127 = reinterpret_tensor(buf123, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf124, arg88_1, buf127, 393216, grid=grid(393216), stream=stream0)
        del arg88_1
        buf128 = reinterpret_tensor(buf124, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf125, arg90_1, buf128, 393216, grid=grid(393216), stream=stream0)
        del arg90_1
        del buf125
        # Source Nodes: [], Original ATen: []
        buf129 = aten._scaled_dot_product_efficient_attention(buf126, buf127, buf128, None, False, scale=0.125)
        buf130 = buf129[0]
        del buf129
        buf134 = reinterpret_tensor(buf128, (512, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (512, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), out=buf134)
        del arg91_1
        buf138 = reinterpret_tensor(buf130, (1, 512, 768), (393216, 768, 1), 0); del buf130  # reuse
        # Source Nodes: [add_19, attention_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf134, arg92_1, buf122, arg93_1, arg94_1, buf138, 512, 768, grid=grid(512), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        buf139 = reinterpret_tensor(buf117, (512, 3072), (3072, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (512, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 3072), (1, 768), 0), out=buf139)
        del arg95_1
        buf140 = reinterpret_tensor(buf139, (1, 512, 3072), (1572864, 3072, 1), 0); del buf139  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf140, arg96_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg96_1
        buf141 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg97_1, (3072, 768), (1, 3072), 0), out=buf141)
        del arg97_1
        buf145 = buf122; del buf122  # reuse
        # Source Nodes: [add_20, hidden_states_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf141, arg98_1, buf138, arg99_1, arg100_1, buf145, 512, 768, grid=grid(512), stream=stream0)
        del arg100_1
        del arg98_1
        del arg99_1
        buf146 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (512, 768), (768, 1), 0), reinterpret_tensor(arg101_1, (768, 768), (1, 768), 0), out=buf146)
        del arg101_1
        buf147 = reinterpret_tensor(buf138, (512, 768), (768, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (512, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 768), (1, 768), 0), out=buf147)
        del arg103_1
        buf148 = reinterpret_tensor(buf127, (512, 768), (768, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf148)
        del arg105_1
        buf149 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf146, arg102_1, buf149, 393216, grid=grid(393216), stream=stream0)
        del arg102_1
        buf150 = reinterpret_tensor(buf146, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf147, arg104_1, buf150, 393216, grid=grid(393216), stream=stream0)
        del arg104_1
        buf151 = reinterpret_tensor(buf147, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf148, arg106_1, buf151, 393216, grid=grid(393216), stream=stream0)
        del arg106_1
        del buf148
        # Source Nodes: [], Original ATen: []
        buf152 = aten._scaled_dot_product_efficient_attention(buf149, buf150, buf151, None, False, scale=0.125)
        buf153 = buf152[0]
        del buf152
        buf157 = reinterpret_tensor(buf151, (512, 768), (768, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), out=buf157)
        del arg107_1
        buf161 = reinterpret_tensor(buf153, (1, 512, 768), (393216, 768, 1), 0); del buf153  # reuse
        # Source Nodes: [add_22, attention_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf157, arg108_1, buf145, arg109_1, arg110_1, buf161, 512, 768, grid=grid(512), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        buf162 = reinterpret_tensor(buf140, (512, 3072), (3072, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 3072), (1, 768), 0), out=buf162)
        del arg111_1
        buf163 = reinterpret_tensor(buf162, (1, 512, 3072), (1572864, 3072, 1), 0); del buf162  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf163, arg112_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg112_1
        buf164 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf163, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg113_1, (3072, 768), (1, 3072), 0), out=buf164)
        del arg113_1
        buf168 = buf145; del buf145  # reuse
        # Source Nodes: [add_23, hidden_states_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf164, arg114_1, buf161, arg115_1, arg116_1, buf168, 512, 768, grid=grid(512), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        buf169 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (512, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf169)
        del arg117_1
        buf170 = reinterpret_tensor(buf161, (512, 768), (768, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (512, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), out=buf170)
        del arg119_1
        buf171 = reinterpret_tensor(buf150, (512, 768), (768, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (512, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), out=buf171)
        del arg121_1
        buf172 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf169, arg118_1, buf172, 393216, grid=grid(393216), stream=stream0)
        del arg118_1
        buf173 = reinterpret_tensor(buf169, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf170, arg120_1, buf173, 393216, grid=grid(393216), stream=stream0)
        del arg120_1
        buf174 = reinterpret_tensor(buf170, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf171, arg122_1, buf174, 393216, grid=grid(393216), stream=stream0)
        del arg122_1
        del buf171
        # Source Nodes: [], Original ATen: []
        buf175 = aten._scaled_dot_product_efficient_attention(buf172, buf173, buf174, None, False, scale=0.125)
        buf176 = buf175[0]
        del buf175
        buf180 = reinterpret_tensor(buf174, (512, 768), (768, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (512, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), out=buf180)
        del arg123_1
        buf184 = reinterpret_tensor(buf176, (1, 512, 768), (393216, 768, 1), 0); del buf176  # reuse
        # Source Nodes: [add_25, attention_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf180, arg124_1, buf168, arg125_1, arg126_1, buf184, 512, 768, grid=grid(512), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        buf185 = reinterpret_tensor(buf163, (512, 3072), (3072, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (512, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 3072), (1, 768), 0), out=buf185)
        del arg127_1
        buf186 = reinterpret_tensor(buf185, (1, 512, 3072), (1572864, 3072, 1), 0); del buf185  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf186, arg128_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg128_1
        buf187 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg129_1, (3072, 768), (1, 3072), 0), out=buf187)
        del arg129_1
        buf191 = buf168; del buf168  # reuse
        # Source Nodes: [add_26, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf187, arg130_1, buf184, arg131_1, arg132_1, buf191, 512, 768, grid=grid(512), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        buf192 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (512, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 768), (1, 768), 0), out=buf192)
        del arg133_1
        buf193 = reinterpret_tensor(buf184, (512, 768), (768, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (512, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 768), (1, 768), 0), out=buf193)
        del arg135_1
        buf194 = reinterpret_tensor(buf173, (512, 768), (768, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (512, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), out=buf194)
        del arg137_1
        buf195 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf192, arg134_1, buf195, 393216, grid=grid(393216), stream=stream0)
        del arg134_1
        buf196 = reinterpret_tensor(buf192, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf193, arg136_1, buf196, 393216, grid=grid(393216), stream=stream0)
        del arg136_1
        buf197 = reinterpret_tensor(buf193, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf194, arg138_1, buf197, 393216, grid=grid(393216), stream=stream0)
        del arg138_1
        del buf194
        # Source Nodes: [], Original ATen: []
        buf198 = aten._scaled_dot_product_efficient_attention(buf195, buf196, buf197, None, False, scale=0.125)
        buf199 = buf198[0]
        del buf198
        buf203 = reinterpret_tensor(buf197, (512, 768), (768, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (512, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), out=buf203)
        del arg139_1
        buf207 = reinterpret_tensor(buf199, (1, 512, 768), (393216, 768, 1), 0); del buf199  # reuse
        # Source Nodes: [add_28, attention_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf203, arg140_1, buf191, arg141_1, arg142_1, buf207, 512, 768, grid=grid(512), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        buf208 = reinterpret_tensor(buf186, (512, 3072), (3072, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 3072), (1, 768), 0), out=buf208)
        del arg143_1
        buf209 = reinterpret_tensor(buf208, (1, 512, 3072), (1572864, 3072, 1), 0); del buf208  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf209, arg144_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg144_1
        buf210 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg145_1, (3072, 768), (1, 3072), 0), out=buf210)
        del arg145_1
        buf214 = buf191; del buf191  # reuse
        # Source Nodes: [add_29, hidden_states_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf210, arg146_1, buf207, arg147_1, arg148_1, buf214, 512, 768, grid=grid(512), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        buf215 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 768), (1, 768), 0), out=buf215)
        del arg149_1
        buf216 = reinterpret_tensor(buf207, (512, 768), (768, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 768), (1, 768), 0), out=buf216)
        del arg151_1
        buf217 = reinterpret_tensor(buf196, (512, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), out=buf217)
        del arg153_1
        buf218 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf215, arg150_1, buf218, 393216, grid=grid(393216), stream=stream0)
        del arg150_1
        buf219 = reinterpret_tensor(buf215, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf216, arg152_1, buf219, 393216, grid=grid(393216), stream=stream0)
        del arg152_1
        buf220 = reinterpret_tensor(buf216, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf217, arg154_1, buf220, 393216, grid=grid(393216), stream=stream0)
        del arg154_1
        del buf217
        # Source Nodes: [], Original ATen: []
        buf221 = aten._scaled_dot_product_efficient_attention(buf218, buf219, buf220, None, False, scale=0.125)
        buf222 = buf221[0]
        del buf221
        buf226 = reinterpret_tensor(buf220, (512, 768), (768, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (512, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), out=buf226)
        del arg155_1
        buf230 = reinterpret_tensor(buf222, (1, 512, 768), (393216, 768, 1), 0); del buf222  # reuse
        # Source Nodes: [add_31, attention_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf226, arg156_1, buf214, arg157_1, arg158_1, buf230, 512, 768, grid=grid(512), stream=stream0)
        del arg156_1
        del arg157_1
        del arg158_1
        buf231 = reinterpret_tensor(buf209, (512, 3072), (3072, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (512, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 3072), (1, 768), 0), out=buf231)
        del arg159_1
        buf232 = reinterpret_tensor(buf231, (1, 512, 3072), (1572864, 3072, 1), 0); del buf231  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf232, arg160_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg160_1
        buf233 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg161_1, (3072, 768), (1, 3072), 0), out=buf233)
        del arg161_1
        buf237 = buf214; del buf214  # reuse
        # Source Nodes: [add_32, hidden_states_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf233, arg162_1, buf230, arg163_1, arg164_1, buf237, 512, 768, grid=grid(512), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        buf238 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 768), (1, 768), 0), out=buf238)
        del arg165_1
        buf239 = reinterpret_tensor(buf230, (512, 768), (768, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), out=buf239)
        del arg167_1
        buf240 = reinterpret_tensor(buf219, (512, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), out=buf240)
        del arg169_1
        buf241 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf238, arg166_1, buf241, 393216, grid=grid(393216), stream=stream0)
        del arg166_1
        buf242 = reinterpret_tensor(buf238, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf239, arg168_1, buf242, 393216, grid=grid(393216), stream=stream0)
        del arg168_1
        buf243 = reinterpret_tensor(buf239, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf240, arg170_1, buf243, 393216, grid=grid(393216), stream=stream0)
        del arg170_1
        del buf240
        # Source Nodes: [], Original ATen: []
        buf244 = aten._scaled_dot_product_efficient_attention(buf241, buf242, buf243, None, False, scale=0.125)
        buf245 = buf244[0]
        del buf244
        buf249 = reinterpret_tensor(buf243, (512, 768), (768, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (512, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), out=buf249)
        del arg171_1
        buf253 = reinterpret_tensor(buf245, (1, 512, 768), (393216, 768, 1), 0); del buf245  # reuse
        # Source Nodes: [add_34, attention_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf249, arg172_1, buf237, arg173_1, arg174_1, buf253, 512, 768, grid=grid(512), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        buf254 = reinterpret_tensor(buf232, (512, 3072), (3072, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 3072), (1, 768), 0), out=buf254)
        del arg175_1
        buf255 = reinterpret_tensor(buf254, (1, 512, 3072), (1572864, 3072, 1), 0); del buf254  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf255, arg176_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg176_1
        buf256 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg177_1, (3072, 768), (1, 3072), 0), out=buf256)
        del arg177_1
        buf260 = buf237; del buf237  # reuse
        # Source Nodes: [add_35, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf256, arg178_1, buf253, arg179_1, arg180_1, buf260, 512, 768, grid=grid(512), stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        buf261 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 768), (1, 768), 0), out=buf261)
        del arg181_1
        buf262 = reinterpret_tensor(buf253, (512, 768), (768, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), out=buf262)
        del arg183_1
        buf263 = reinterpret_tensor(buf242, (512, 768), (768, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), out=buf263)
        del arg185_1
        buf264 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf261, arg182_1, buf264, 393216, grid=grid(393216), stream=stream0)
        del arg182_1
        buf265 = reinterpret_tensor(buf261, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf262, arg184_1, buf265, 393216, grid=grid(393216), stream=stream0)
        del arg184_1
        buf266 = reinterpret_tensor(buf262, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf263, arg186_1, buf266, 393216, grid=grid(393216), stream=stream0)
        del arg186_1
        del buf263
        # Source Nodes: [], Original ATen: []
        buf267 = aten._scaled_dot_product_efficient_attention(buf264, buf265, buf266, None, False, scale=0.125)
        del buf264
        del buf265
        buf268 = buf267[0]
        del buf267
        buf272 = reinterpret_tensor(buf266, (512, 768), (768, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (512, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), out=buf272)
        del arg187_1
        buf276 = reinterpret_tensor(buf268, (1, 512, 768), (393216, 768, 1), 0); del buf268  # reuse
        # Source Nodes: [add_37, attention_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf272, arg188_1, buf260, arg189_1, arg190_1, buf276, 512, 768, grid=grid(512), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        buf277 = reinterpret_tensor(buf255, (512, 3072), (3072, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 3072), (1, 768), 0), out=buf277)
        del arg191_1
        buf278 = reinterpret_tensor(buf277, (1, 512, 3072), (1572864, 3072, 1), 0); del buf277  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf278, arg192_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg192_1
        buf279 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg193_1, (3072, 768), (1, 3072), 0), out=buf279)
        del arg193_1
        del buf278
        buf283 = buf260; del buf260  # reuse
        # Source Nodes: [add_38, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf279, arg194_1, buf276, arg195_1, arg196_1, buf283, 512, 768, grid=grid(512), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        del buf276
        buf284 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf283, (512, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 768), (1, 768), 0), out=buf284)
        del arg197_1
        buf288 = buf283; del buf283  # reuse
        # Source Nodes: [x_37, x_38], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_5.run(buf284, arg198_1, arg199_1, arg200_1, buf288, 512, 768, grid=grid(512), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del buf284
        buf289 = empty((512, 32005), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg202_1, reinterpret_tensor(buf288, (512, 768), (768, 1), 0), reinterpret_tensor(arg201_1, (768, 32005), (1, 768), 0), alpha=1, beta=1, out=buf289)
        del arg201_1
        del arg202_1
        del buf288
        buf290 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        buf291 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_6.run(buf289, buf290, buf291, 512, 32005, grid=grid(512), stream=stream0)
        buf292 = empty((), device='cuda', dtype=torch.float32)
        buf294 = buf292; del buf292  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_7.run(buf294, arg205_1, buf289, buf290, buf291, 1, 512, grid=grid(1), stream=stream0)
        del arg205_1
        return (buf294, reinterpret_tensor(buf289, (1, 512, 32005), (16386560, 32005, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((514, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((32005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((32005, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1, 514), (514, 1), device='cuda:0', dtype=torch.int64)
    arg204_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg205_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('CamemBert', benchmark_compiled_module)
