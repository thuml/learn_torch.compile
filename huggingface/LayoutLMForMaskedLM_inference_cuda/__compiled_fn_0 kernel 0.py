
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


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzz5ehtwj35tbwkobirwj53dxxby36hbs3eksws4fw4b33bzkzr.py
# Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, embeddings, embeddings_1, h_position_embeddings, left_position_embeddings, lower_position_embeddings, position_embeddings, right_position_embeddings, sub_1, sub_2, token_type_embeddings, token_type_ids, upper_position_embeddings, w_position_embeddings, words_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.sub, aten.zeros]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# add_4 => add_4
# add_5 => add_5
# add_6 => add_6
# embeddings => add_7
# embeddings_1 => add_8, add_9, mul_1, mul_2, rsqrt, sub_3, var_mean
# h_position_embeddings => embedding_6
# left_position_embeddings => embedding_2
# lower_position_embeddings => embedding_5
# position_embeddings => embedding_1
# right_position_embeddings => embedding_4
# sub_1 => full_default_2
# sub_2 => full_default_3
# token_type_embeddings => embedding_8
# token_type_ids => full_default
# upper_position_embeddings => embedding_3
# w_position_embeddings => embedding_7
# words_embeddings => embedding
triton_per_fused_add_embedding_native_layer_norm_sub_zeros_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_sub_zeros_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp11 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 512
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 512)) | ~xmask, "index out of bounds: 0 <= tmp8 < 512")
    tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 + tmp11
    tmp16 = tmp15 + tmp13
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
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
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp18, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp49, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2s3ic7nygtbrrgdj5txjepzrncknc7p3bsvfeu5pcyzyis6bmb.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jx/cjxmpxd7r3nj75hmevjg5cbsdprzcpebwdrthtnojziqqcnestyt.py
# Source Nodes: [add_9, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_9 => add_11
# attention_output => add_12, add_13, mul_3, mul_4, rsqrt_1, sub_5, var_mean_1
triton_per_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_2', 'mutated_arg_names': []}
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
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cykpgcexpqolkloidvb3xedkxdgxpplc33pcx23f32oonp6jqso5.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_14, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/lo/clopg6iqmuaja4ja4lprlc6qffkvaoyaibval57yja63rq43viah.py
# Source Nodes: [hidden_states_109, hidden_states_111], Original ATen: [aten.gelu, aten.native_layer_norm]
# hidden_states_109 => add_106, erf_12, mul_87, mul_88, mul_89
# hidden_states_111 => add_107, add_108, mul_90, mul_91, rsqrt_25, sub_40, var_mean_25
triton_per_fused_gelu_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_4', 'mutated_arg_names': []}
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
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/cojvao7rxqr33qwnkejkms72skx3egifpbej237msmjtcwyoocbg.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
# masked_lm_loss => amax_12, exp_12, sub_41, sum_13
triton_red_fused__log_softmax_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4ikfuk62ozrlcl7ygdi43zajegro4ac3r335rpwgdzpmy3cjfs.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# masked_lm_loss => convert_element_type, div_24, full_default_5, ne_1, ne_2, neg, sum_14, sum_15, where_1
triton_per_fused_nll_loss_forward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_6', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = tmp4 + 30522
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 30522), "index out of bounds: 0 <= tmp7 < 30522")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (30522*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30522, 768), (768, 1))
    assert_size_stride(arg1_1, (512, 768), (768, 1))
    assert_size_stride(arg2_1, (1024, 768), (768, 1))
    assert_size_stride(arg3_1, (1024, 768), (768, 1))
    assert_size_stride(arg4_1, (1024, 768), (768, 1))
    assert_size_stride(arg5_1, (1024, 768), (768, 1))
    assert_size_stride(arg6_1, (2, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, 768), (768, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, 768), (768, 1))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (3072, 768), (768, 1))
    assert_size_stride(arg20_1, (3072, ), (1, ))
    assert_size_stride(arg21_1, (768, 3072), (3072, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, 768), (768, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, 768), (768, 1))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (3072, 768), (768, 1))
    assert_size_stride(arg36_1, (3072, ), (1, ))
    assert_size_stride(arg37_1, (768, 3072), (3072, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, 768), (768, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, 768), (768, 1))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (3072, 768), (768, 1))
    assert_size_stride(arg52_1, (3072, ), (1, ))
    assert_size_stride(arg53_1, (768, 3072), (3072, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, 768), (768, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, 768), (768, 1))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (3072, 768), (768, 1))
    assert_size_stride(arg68_1, (3072, ), (1, ))
    assert_size_stride(arg69_1, (768, 3072), (3072, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, 768), (768, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, 768), (768, 1))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (3072, 768), (768, 1))
    assert_size_stride(arg84_1, (3072, ), (1, ))
    assert_size_stride(arg85_1, (768, 3072), (3072, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, 768), (768, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, 768), (768, 1))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (3072, 768), (768, 1))
    assert_size_stride(arg100_1, (3072, ), (1, ))
    assert_size_stride(arg101_1, (768, 3072), (3072, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, 768), (768, 1))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, 768), (768, 1))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (3072, 768), (768, 1))
    assert_size_stride(arg116_1, (3072, ), (1, ))
    assert_size_stride(arg117_1, (768, 3072), (3072, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, 768), (768, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, 768), (768, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, 768), (768, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, 768), (768, 1))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (3072, 768), (768, 1))
    assert_size_stride(arg132_1, (3072, ), (1, ))
    assert_size_stride(arg133_1, (768, 3072), (3072, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, 768), (768, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, 768), (768, 1))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (3072, 768), (768, 1))
    assert_size_stride(arg148_1, (3072, ), (1, ))
    assert_size_stride(arg149_1, (768, 3072), (3072, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, 768), (768, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, 768), (768, 1))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, 768), (768, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (3072, 768), (768, 1))
    assert_size_stride(arg164_1, (3072, ), (1, ))
    assert_size_stride(arg165_1, (768, 3072), (3072, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, 768), (768, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, 768), (768, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, 768), (768, 1))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, 768), (768, 1))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (3072, 768), (768, 1))
    assert_size_stride(arg180_1, (3072, ), (1, ))
    assert_size_stride(arg181_1, (768, 3072), (3072, 1))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, 768), (768, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, 768), (768, 1))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, 768), (768, 1))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (3072, 768), (768, 1))
    assert_size_stride(arg196_1, (3072, ), (1, ))
    assert_size_stride(arg197_1, (768, 3072), (3072, 1))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, 768), (768, 1))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (768, 768), (768, 1))
    assert_size_stride(arg204_1, (768, ), (1, ))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (30522, 768), (768, 1))
    assert_size_stride(arg208_1, (30522, ), (1, ))
    assert_size_stride(arg209_1, (1, 512), (512, 1))
    assert_size_stride(arg210_1, (1, 512), (512, 1))
    assert_size_stride(arg211_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, embeddings, embeddings_1, h_position_embeddings, left_position_embeddings, lower_position_embeddings, position_embeddings, right_position_embeddings, sub_1, sub_2, token_type_embeddings, token_type_ids, upper_position_embeddings, w_position_embeddings, words_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.sub, aten.zeros]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_sub_zeros_0.run(arg210_1, arg0_1, arg209_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, buf0, buf4, 512, 768, grid=grid(512), stream=stream0)
        del arg0_1
        del arg1_1
        del arg209_1
        del arg210_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        buf5 = reinterpret_tensor(buf0, (512, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf5)
        del arg9_1
        buf6 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), out=buf6)
        del arg11_1
        buf7 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf7)
        del arg13_1
        buf8 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf5, arg10_1, buf8, 393216, grid=grid(393216), stream=stream0)
        del arg10_1
        buf9 = reinterpret_tensor(buf5, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf6, arg12_1, buf9, 393216, grid=grid(393216), stream=stream0)
        del arg12_1
        buf10 = reinterpret_tensor(buf6, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf7, arg14_1, buf10, 393216, grid=grid(393216), stream=stream0)
        del arg14_1
        del buf7
        # Source Nodes: [], Original ATen: []
        buf11 = aten._scaled_dot_product_efficient_attention(buf8, buf9, buf10, None, False, scale=0.125)
        buf12 = buf11[0]
        del buf11
        buf16 = reinterpret_tensor(buf9, (512, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 768), (1, 768), 0), out=buf16)
        del arg15_1
        buf20 = reinterpret_tensor(buf12, (1, 512, 768), (393216, 768, 1), 0); del buf12  # reuse
        # Source Nodes: [add_9, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf16, arg16_1, buf4, arg17_1, arg18_1, buf20, 512, 768, grid=grid(512), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        buf21 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 3072), (1, 768), 0), out=buf21)
        del arg19_1
        buf22 = reinterpret_tensor(buf21, (1, 512, 3072), (1572864, 3072, 1), 0); del buf21  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf22, arg20_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg20_1
        buf23 = reinterpret_tensor(buf4, (512, 768), (768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg21_1, (3072, 768), (1, 3072), 0), out=buf23)
        del arg21_1
        buf27 = reinterpret_tensor(buf16, (1, 512, 768), (393216, 768, 1), 0); del buf16  # reuse
        # Source Nodes: [add_10, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf23, arg22_1, buf20, arg23_1, arg24_1, buf27, 512, 768, grid=grid(512), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), out=buf28)
        del arg25_1
        buf29 = reinterpret_tensor(buf20, (512, 768), (768, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), out=buf29)
        del arg27_1
        buf30 = reinterpret_tensor(buf8, (512, 768), (768, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), out=buf30)
        del arg29_1
        buf31 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf28, arg26_1, buf31, 393216, grid=grid(393216), stream=stream0)
        del arg26_1
        buf32 = reinterpret_tensor(buf28, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf29, arg28_1, buf32, 393216, grid=grid(393216), stream=stream0)
        del arg28_1
        buf33 = reinterpret_tensor(buf29, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf30, arg30_1, buf33, 393216, grid=grid(393216), stream=stream0)
        del arg30_1
        del buf30
        # Source Nodes: [], Original ATen: []
        buf34 = aten._scaled_dot_product_efficient_attention(buf31, buf32, buf33, None, False, scale=0.125)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf33, (512, 768), (768, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (512, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 768), (1, 768), 0), out=buf39)
        del arg31_1
        buf43 = reinterpret_tensor(buf35, (1, 512, 768), (393216, 768, 1), 0); del buf35  # reuse
        # Source Nodes: [add_12, attention_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf39, arg32_1, buf27, arg33_1, arg34_1, buf43, 512, 768, grid=grid(512), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf44 = reinterpret_tensor(buf22, (512, 3072), (3072, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf43, (512, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 3072), (1, 768), 0), out=buf44)
        del arg35_1
        buf45 = reinterpret_tensor(buf44, (1, 512, 3072), (1572864, 3072, 1), 0); del buf44  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf45, arg36_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg36_1
        buf46 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg37_1, (3072, 768), (1, 3072), 0), out=buf46)
        del arg37_1
        buf50 = buf27; del buf27  # reuse
        # Source Nodes: [add_13, hidden_states_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf46, arg38_1, buf43, arg39_1, arg40_1, buf50, 512, 768, grid=grid(512), stream=stream0)
        del arg38_1
        del arg39_1
        del arg40_1
        buf51 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (512, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), out=buf51)
        del arg41_1
        buf52 = reinterpret_tensor(buf43, (512, 768), (768, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (512, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), out=buf52)
        del arg43_1
        buf53 = reinterpret_tensor(buf32, (512, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (512, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), out=buf53)
        del arg45_1
        buf54 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf51, arg42_1, buf54, 393216, grid=grid(393216), stream=stream0)
        del arg42_1
        buf55 = reinterpret_tensor(buf51, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf52, arg44_1, buf55, 393216, grid=grid(393216), stream=stream0)
        del arg44_1
        buf56 = reinterpret_tensor(buf52, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf53, arg46_1, buf56, 393216, grid=grid(393216), stream=stream0)
        del arg46_1
        del buf53
        # Source Nodes: [], Original ATen: []
        buf57 = aten._scaled_dot_product_efficient_attention(buf54, buf55, buf56, None, False, scale=0.125)
        buf58 = buf57[0]
        del buf57
        buf62 = reinterpret_tensor(buf56, (512, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (512, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 768), (1, 768), 0), out=buf62)
        del arg47_1
        buf66 = reinterpret_tensor(buf58, (1, 512, 768), (393216, 768, 1), 0); del buf58  # reuse
        # Source Nodes: [add_15, attention_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf62, arg48_1, buf50, arg49_1, arg50_1, buf66, 512, 768, grid=grid(512), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        buf67 = reinterpret_tensor(buf45, (512, 3072), (3072, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (512, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 3072), (1, 768), 0), out=buf67)
        del arg51_1
        buf68 = reinterpret_tensor(buf67, (1, 512, 3072), (1572864, 3072, 1), 0); del buf67  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf68, arg52_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg52_1
        buf69 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg53_1, (3072, 768), (1, 3072), 0), out=buf69)
        del arg53_1
        buf73 = buf50; del buf50  # reuse
        # Source Nodes: [add_16, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf69, arg54_1, buf66, arg55_1, arg56_1, buf73, 512, 768, grid=grid(512), stream=stream0)
        del arg54_1
        del arg55_1
        del arg56_1
        buf74 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (512, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf74)
        del arg57_1
        buf75 = reinterpret_tensor(buf66, (512, 768), (768, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (512, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), out=buf75)
        del arg59_1
        buf76 = reinterpret_tensor(buf55, (512, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (512, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 768), (1, 768), 0), out=buf76)
        del arg61_1
        buf77 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf74, arg58_1, buf77, 393216, grid=grid(393216), stream=stream0)
        del arg58_1
        buf78 = reinterpret_tensor(buf74, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf75, arg60_1, buf78, 393216, grid=grid(393216), stream=stream0)
        del arg60_1
        buf79 = reinterpret_tensor(buf75, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf76, arg62_1, buf79, 393216, grid=grid(393216), stream=stream0)
        del arg62_1
        del buf76
        # Source Nodes: [], Original ATen: []
        buf80 = aten._scaled_dot_product_efficient_attention(buf77, buf78, buf79, None, False, scale=0.125)
        buf81 = buf80[0]
        del buf80
        buf85 = reinterpret_tensor(buf79, (512, 768), (768, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (512, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 768), (1, 768), 0), out=buf85)
        del arg63_1
        buf89 = reinterpret_tensor(buf81, (1, 512, 768), (393216, 768, 1), 0); del buf81  # reuse
        # Source Nodes: [add_18, attention_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf85, arg64_1, buf73, arg65_1, arg66_1, buf89, 512, 768, grid=grid(512), stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        buf90 = reinterpret_tensor(buf68, (512, 3072), (3072, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf89, (512, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 3072), (1, 768), 0), out=buf90)
        del arg67_1
        buf91 = reinterpret_tensor(buf90, (1, 512, 3072), (1572864, 3072, 1), 0); del buf90  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf91, arg68_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg68_1
        buf92 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg69_1, (3072, 768), (1, 3072), 0), out=buf92)
        del arg69_1
        buf96 = buf73; del buf73  # reuse
        # Source Nodes: [add_19, hidden_states_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf92, arg70_1, buf89, arg71_1, arg72_1, buf96, 512, 768, grid=grid(512), stream=stream0)
        del arg70_1
        del arg71_1
        del arg72_1
        buf97 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (512, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), out=buf97)
        del arg73_1
        buf98 = reinterpret_tensor(buf89, (512, 768), (768, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (512, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), out=buf98)
        del arg75_1
        buf99 = reinterpret_tensor(buf78, (512, 768), (768, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (512, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), out=buf99)
        del arg77_1
        buf100 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf97, arg74_1, buf100, 393216, grid=grid(393216), stream=stream0)
        del arg74_1
        buf101 = reinterpret_tensor(buf97, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf98, arg76_1, buf101, 393216, grid=grid(393216), stream=stream0)
        del arg76_1
        buf102 = reinterpret_tensor(buf98, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf99, arg78_1, buf102, 393216, grid=grid(393216), stream=stream0)
        del arg78_1
        del buf99
        # Source Nodes: [], Original ATen: []
        buf103 = aten._scaled_dot_product_efficient_attention(buf100, buf101, buf102, None, False, scale=0.125)
        buf104 = buf103[0]
        del buf103
        buf108 = reinterpret_tensor(buf102, (512, 768), (768, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (512, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 768), (1, 768), 0), out=buf108)
        del arg79_1
        buf112 = reinterpret_tensor(buf104, (1, 512, 768), (393216, 768, 1), 0); del buf104  # reuse
        # Source Nodes: [add_21, attention_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf108, arg80_1, buf96, arg81_1, arg82_1, buf112, 512, 768, grid=grid(512), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        buf113 = reinterpret_tensor(buf91, (512, 3072), (3072, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf112, (512, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 3072), (1, 768), 0), out=buf113)
        del arg83_1
        buf114 = reinterpret_tensor(buf113, (1, 512, 3072), (1572864, 3072, 1), 0); del buf113  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf114, arg84_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg84_1
        buf115 = reinterpret_tensor(buf96, (512, 768), (768, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg85_1, (3072, 768), (1, 3072), 0), out=buf115)
        del arg85_1
        buf119 = reinterpret_tensor(buf108, (1, 512, 768), (393216, 768, 1), 0); del buf108  # reuse
        # Source Nodes: [add_22, hidden_states_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf115, arg86_1, buf112, arg87_1, arg88_1, buf119, 512, 768, grid=grid(512), stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        buf120 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (512, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), out=buf120)
        del arg89_1
        buf121 = reinterpret_tensor(buf112, (512, 768), (768, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (512, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), out=buf121)
        del arg91_1
        buf122 = reinterpret_tensor(buf101, (512, 768), (768, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (512, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), out=buf122)
        del arg93_1
        buf123 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf120, arg90_1, buf123, 393216, grid=grid(393216), stream=stream0)
        del arg90_1
        buf124 = reinterpret_tensor(buf120, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf121, arg92_1, buf124, 393216, grid=grid(393216), stream=stream0)
        del arg92_1
        buf125 = reinterpret_tensor(buf121, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf122, arg94_1, buf125, 393216, grid=grid(393216), stream=stream0)
        del arg94_1
        del buf122
        # Source Nodes: [], Original ATen: []
        buf126 = aten._scaled_dot_product_efficient_attention(buf123, buf124, buf125, None, False, scale=0.125)
        buf127 = buf126[0]
        del buf126
        buf131 = reinterpret_tensor(buf125, (512, 768), (768, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (512, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 768), (1, 768), 0), out=buf131)
        del arg95_1
        buf135 = reinterpret_tensor(buf127, (1, 512, 768), (393216, 768, 1), 0); del buf127  # reuse
        # Source Nodes: [add_24, attention_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf131, arg96_1, buf119, arg97_1, arg98_1, buf135, 512, 768, grid=grid(512), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        buf136 = reinterpret_tensor(buf114, (512, 3072), (3072, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (512, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 3072), (1, 768), 0), out=buf136)
        del arg99_1
        buf137 = reinterpret_tensor(buf136, (1, 512, 3072), (1572864, 3072, 1), 0); del buf136  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf137, arg100_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg100_1
        buf138 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg101_1, (3072, 768), (1, 3072), 0), out=buf138)
        del arg101_1
        buf142 = buf119; del buf119  # reuse
        # Source Nodes: [add_25, hidden_states_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf138, arg102_1, buf135, arg103_1, arg104_1, buf142, 512, 768, grid=grid(512), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        buf143 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf143)
        del arg105_1
        buf144 = reinterpret_tensor(buf135, (512, 768), (768, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (512, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), out=buf144)
        del arg107_1
        buf145 = reinterpret_tensor(buf124, (512, 768), (768, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (512, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 768), (1, 768), 0), out=buf145)
        del arg109_1
        buf146 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf143, arg106_1, buf146, 393216, grid=grid(393216), stream=stream0)
        del arg106_1
        buf147 = reinterpret_tensor(buf143, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf144, arg108_1, buf147, 393216, grid=grid(393216), stream=stream0)
        del arg108_1
        buf148 = reinterpret_tensor(buf144, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf145, arg110_1, buf148, 393216, grid=grid(393216), stream=stream0)
        del arg110_1
        del buf145
        # Source Nodes: [], Original ATen: []
        buf149 = aten._scaled_dot_product_efficient_attention(buf146, buf147, buf148, None, False, scale=0.125)
        buf150 = buf149[0]
        del buf149
        buf154 = reinterpret_tensor(buf148, (512, 768), (768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (512, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 768), (1, 768), 0), out=buf154)
        del arg111_1
        buf158 = reinterpret_tensor(buf150, (1, 512, 768), (393216, 768, 1), 0); del buf150  # reuse
        # Source Nodes: [add_27, attention_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf154, arg112_1, buf142, arg113_1, arg114_1, buf158, 512, 768, grid=grid(512), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        buf159 = reinterpret_tensor(buf137, (512, 3072), (3072, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (512, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 3072), (1, 768), 0), out=buf159)
        del arg115_1
        buf160 = reinterpret_tensor(buf159, (1, 512, 3072), (1572864, 3072, 1), 0); del buf159  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf160, arg116_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg116_1
        buf161 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg117_1, (3072, 768), (1, 3072), 0), out=buf161)
        del arg117_1
        buf165 = buf142; del buf142  # reuse
        # Source Nodes: [add_28, hidden_states_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf161, arg118_1, buf158, arg119_1, arg120_1, buf165, 512, 768, grid=grid(512), stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        buf166 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), out=buf166)
        del arg121_1
        buf167 = reinterpret_tensor(buf158, (512, 768), (768, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), out=buf167)
        del arg123_1
        buf168 = reinterpret_tensor(buf147, (512, 768), (768, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), out=buf168)
        del arg125_1
        buf169 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf166, arg122_1, buf169, 393216, grid=grid(393216), stream=stream0)
        del arg122_1
        buf170 = reinterpret_tensor(buf166, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf167, arg124_1, buf170, 393216, grid=grid(393216), stream=stream0)
        del arg124_1
        buf171 = reinterpret_tensor(buf167, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf168, arg126_1, buf171, 393216, grid=grid(393216), stream=stream0)
        del arg126_1
        del buf168
        # Source Nodes: [], Original ATen: []
        buf172 = aten._scaled_dot_product_efficient_attention(buf169, buf170, buf171, None, False, scale=0.125)
        buf173 = buf172[0]
        del buf172
        buf177 = reinterpret_tensor(buf171, (512, 768), (768, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (512, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 768), (1, 768), 0), out=buf177)
        del arg127_1
        buf181 = reinterpret_tensor(buf173, (1, 512, 768), (393216, 768, 1), 0); del buf173  # reuse
        # Source Nodes: [add_30, attention_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf177, arg128_1, buf165, arg129_1, arg130_1, buf181, 512, 768, grid=grid(512), stream=stream0)
        del arg128_1
        del arg129_1
        del arg130_1
        buf182 = reinterpret_tensor(buf160, (512, 3072), (3072, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (512, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 3072), (1, 768), 0), out=buf182)
        del arg131_1
        buf183 = reinterpret_tensor(buf182, (1, 512, 3072), (1572864, 3072, 1), 0); del buf182  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf183, arg132_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg132_1
        buf184 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg133_1, (3072, 768), (1, 3072), 0), out=buf184)
        del arg133_1
        buf188 = buf165; del buf165  # reuse
        # Source Nodes: [add_31, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf184, arg134_1, buf181, arg135_1, arg136_1, buf188, 512, 768, grid=grid(512), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        buf189 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (512, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), out=buf189)
        del arg137_1
        buf190 = reinterpret_tensor(buf181, (512, 768), (768, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (512, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), out=buf190)
        del arg139_1
        buf191 = reinterpret_tensor(buf170, (512, 768), (768, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (512, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), out=buf191)
        del arg141_1
        buf192 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf189, arg138_1, buf192, 393216, grid=grid(393216), stream=stream0)
        del arg138_1
        buf193 = reinterpret_tensor(buf189, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf190, arg140_1, buf193, 393216, grid=grid(393216), stream=stream0)
        del arg140_1
        buf194 = reinterpret_tensor(buf190, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf191, arg142_1, buf194, 393216, grid=grid(393216), stream=stream0)
        del arg142_1
        del buf191
        # Source Nodes: [], Original ATen: []
        buf195 = aten._scaled_dot_product_efficient_attention(buf192, buf193, buf194, None, False, scale=0.125)
        buf196 = buf195[0]
        del buf195
        buf200 = reinterpret_tensor(buf194, (512, 768), (768, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), out=buf200)
        del arg143_1
        buf204 = reinterpret_tensor(buf196, (1, 512, 768), (393216, 768, 1), 0); del buf196  # reuse
        # Source Nodes: [add_33, attention_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf200, arg144_1, buf188, arg145_1, arg146_1, buf204, 512, 768, grid=grid(512), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        buf205 = reinterpret_tensor(buf183, (512, 3072), (3072, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (512, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 3072), (1, 768), 0), out=buf205)
        del arg147_1
        buf206 = reinterpret_tensor(buf205, (1, 512, 3072), (1572864, 3072, 1), 0); del buf205  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf206, arg148_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg148_1
        buf207 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg149_1, (3072, 768), (1, 3072), 0), out=buf207)
        del arg149_1
        buf211 = buf188; del buf188  # reuse
        # Source Nodes: [add_34, hidden_states_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf207, arg150_1, buf204, arg151_1, arg152_1, buf211, 512, 768, grid=grid(512), stream=stream0)
        del arg150_1
        del arg151_1
        del arg152_1
        buf212 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), out=buf212)
        del arg153_1
        buf213 = reinterpret_tensor(buf204, (512, 768), (768, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), out=buf213)
        del arg155_1
        buf214 = reinterpret_tensor(buf193, (512, 768), (768, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), out=buf214)
        del arg157_1
        buf215 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf212, arg154_1, buf215, 393216, grid=grid(393216), stream=stream0)
        del arg154_1
        buf216 = reinterpret_tensor(buf212, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf213, arg156_1, buf216, 393216, grid=grid(393216), stream=stream0)
        del arg156_1
        buf217 = reinterpret_tensor(buf213, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf214, arg158_1, buf217, 393216, grid=grid(393216), stream=stream0)
        del arg158_1
        del buf214
        # Source Nodes: [], Original ATen: []
        buf218 = aten._scaled_dot_product_efficient_attention(buf215, buf216, buf217, None, False, scale=0.125)
        buf219 = buf218[0]
        del buf218
        buf223 = reinterpret_tensor(buf217, (512, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (512, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 768), (1, 768), 0), out=buf223)
        del arg159_1
        buf227 = reinterpret_tensor(buf219, (1, 512, 768), (393216, 768, 1), 0); del buf219  # reuse
        # Source Nodes: [add_36, attention_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf223, arg160_1, buf211, arg161_1, arg162_1, buf227, 512, 768, grid=grid(512), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        buf228 = reinterpret_tensor(buf206, (512, 3072), (3072, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (512, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 3072), (1, 768), 0), out=buf228)
        del arg163_1
        buf229 = reinterpret_tensor(buf228, (1, 512, 3072), (1572864, 3072, 1), 0); del buf228  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf229, arg164_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg164_1
        buf230 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg165_1, (3072, 768), (1, 3072), 0), out=buf230)
        del arg165_1
        buf234 = buf211; del buf211  # reuse
        # Source Nodes: [add_37, hidden_states_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf230, arg166_1, buf227, arg167_1, arg168_1, buf234, 512, 768, grid=grid(512), stream=stream0)
        del arg166_1
        del arg167_1
        del arg168_1
        buf235 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (512, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), out=buf235)
        del arg169_1
        buf236 = reinterpret_tensor(buf227, (512, 768), (768, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (512, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), out=buf236)
        del arg171_1
        buf237 = reinterpret_tensor(buf216, (512, 768), (768, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (512, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), out=buf237)
        del arg173_1
        buf238 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf235, arg170_1, buf238, 393216, grid=grid(393216), stream=stream0)
        del arg170_1
        buf239 = reinterpret_tensor(buf235, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf236, arg172_1, buf239, 393216, grid=grid(393216), stream=stream0)
        del arg172_1
        buf240 = reinterpret_tensor(buf236, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf237, arg174_1, buf240, 393216, grid=grid(393216), stream=stream0)
        del arg174_1
        del buf237
        # Source Nodes: [], Original ATen: []
        buf241 = aten._scaled_dot_product_efficient_attention(buf238, buf239, buf240, None, False, scale=0.125)
        buf242 = buf241[0]
        del buf241
        buf246 = reinterpret_tensor(buf240, (512, 768), (768, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (512, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 768), (1, 768), 0), out=buf246)
        del arg175_1
        buf250 = reinterpret_tensor(buf242, (1, 512, 768), (393216, 768, 1), 0); del buf242  # reuse
        # Source Nodes: [add_39, attention_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf246, arg176_1, buf234, arg177_1, arg178_1, buf250, 512, 768, grid=grid(512), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        buf251 = reinterpret_tensor(buf229, (512, 3072), (3072, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (512, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 3072), (1, 768), 0), out=buf251)
        del arg179_1
        buf252 = reinterpret_tensor(buf251, (1, 512, 3072), (1572864, 3072, 1), 0); del buf251  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf252, arg180_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg180_1
        buf253 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg181_1, (3072, 768), (1, 3072), 0), out=buf253)
        del arg181_1
        buf257 = buf234; del buf234  # reuse
        # Source Nodes: [add_40, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf253, arg182_1, buf250, arg183_1, arg184_1, buf257, 512, 768, grid=grid(512), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        buf258 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (512, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), out=buf258)
        del arg185_1
        buf259 = reinterpret_tensor(buf250, (512, 768), (768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (512, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), out=buf259)
        del arg187_1
        buf260 = reinterpret_tensor(buf239, (512, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (512, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 768), (1, 768), 0), out=buf260)
        del arg189_1
        buf261 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf258, arg186_1, buf261, 393216, grid=grid(393216), stream=stream0)
        del arg186_1
        buf262 = reinterpret_tensor(buf258, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf259, arg188_1, buf262, 393216, grid=grid(393216), stream=stream0)
        del arg188_1
        buf263 = reinterpret_tensor(buf259, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf260, arg190_1, buf263, 393216, grid=grid(393216), stream=stream0)
        del arg190_1
        del buf260
        # Source Nodes: [], Original ATen: []
        buf264 = aten._scaled_dot_product_efficient_attention(buf261, buf262, buf263, None, False, scale=0.125)
        del buf261
        del buf262
        buf265 = buf264[0]
        del buf264
        buf269 = reinterpret_tensor(buf263, (512, 768), (768, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (512, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 768), (1, 768), 0), out=buf269)
        del arg191_1
        buf273 = reinterpret_tensor(buf265, (1, 512, 768), (393216, 768, 1), 0); del buf265  # reuse
        # Source Nodes: [add_42, attention_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf269, arg192_1, buf257, arg193_1, arg194_1, buf273, 512, 768, grid=grid(512), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        buf274 = reinterpret_tensor(buf252, (512, 3072), (3072, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (512, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 3072), (1, 768), 0), out=buf274)
        del arg195_1
        buf275 = reinterpret_tensor(buf274, (1, 512, 3072), (1572864, 3072, 1), 0); del buf274  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf275, arg196_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg196_1
        buf276 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf275, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg197_1, (3072, 768), (1, 3072), 0), out=buf276)
        del arg197_1
        del buf275
        buf280 = buf257; del buf257  # reuse
        # Source Nodes: [add_43, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf276, arg198_1, buf273, arg199_1, arg200_1, buf280, 512, 768, grid=grid(512), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del buf273
        buf281 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf280, (512, 768), (768, 1), 0), reinterpret_tensor(arg203_1, (768, 768), (1, 768), 0), out=buf281)
        del arg203_1
        buf285 = buf280; del buf280  # reuse
        # Source Nodes: [hidden_states_109, hidden_states_111], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_4.run(buf281, arg204_1, arg205_1, arg206_1, buf285, 512, 768, grid=grid(512), stream=stream0)
        del arg204_1
        del arg205_1
        del arg206_1
        del buf281
        buf286 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg208_1, reinterpret_tensor(buf285, (512, 768), (768, 1), 0), reinterpret_tensor(arg207_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf286)
        del arg207_1
        del arg208_1
        del buf285
        buf287 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        buf288 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_5.run(buf286, buf287, buf288, 512, 30522, grid=grid(512), stream=stream0)
        buf289 = empty((), device='cuda', dtype=torch.float32)
        buf291 = buf289; del buf289  # reuse
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_6.run(buf291, arg211_1, buf286, buf287, buf288, 1, 512, grid=grid(1), stream=stream0)
        del arg211_1
        return (buf291, reinterpret_tensor(buf286, (1, 512, 30522), (15627264, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg210_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg211_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForMaskedLM', benchmark_compiled_module)
