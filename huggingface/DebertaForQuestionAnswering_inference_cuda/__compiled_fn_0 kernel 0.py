
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


# kernel path: /tmp/torchinductor_youkaichao/hg/chgkoqgmesmlhqihlbj5ixu4w2brvtk3osot7x2vtvt3ri7bksvm.py
# Source Nodes: [add, embeddings, embeddings_1, hidden_states_1, mean, mul, position_embeddings, pow_1, query_states, sqrt, sub, sub_1, variance], Original ATen: [aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
# add => add_1
# embeddings => embedding
# embeddings_1 => add
# hidden_states_1 => div
# mean => mean
# mul => mul
# position_embeddings => embedding_1
# pow_1 => pow_1
# query_states => add_2
# sqrt => sqrt
# sub => sub
# sub_1 => sub_1
# variance => mean_1
triton_red_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 + 512
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 512)) | ~xmask, "index out of bounds: 0 <= tmp8 < 512")
        tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp4 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tmp0 + 50265
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp16 < 50265")
        tmp17 = tl.load(in_ptr1 + (r1 + (768*tmp16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp5 + 512
        tmp19 = tmp5 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp5)
        tl.device_assert(((0 <= tmp20) & (tmp20 < 512)) | ~xmask, "index out of bounds: 0 <= tmp20 < 512")
        tmp21 = tl.load(in_ptr3 + (r1 + (768*tmp20)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp17 + tmp21
        tmp23 = 768.0
        tmp24 = tmp12 / tmp23
        tmp25 = tmp22 - tmp24
        tmp26 = tmp25 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp0 + 50265
        tmp32 = tmp0 < 0
        tmp33 = tl.where(tmp32, tmp31, tmp0)
        tl.device_assert(((0 <= tmp33) & (tmp33 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp33 < 50265")
        tmp34 = tl.load(in_ptr1 + (r1 + (768*tmp33)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tmp5 + 512
        tmp36 = tmp5 < 0
        tmp37 = tl.where(tmp36, tmp35, tmp5)
        tl.device_assert(((0 <= tmp37) & (tmp37 < 512)) | ~xmask, "index out of bounds: 0 <= tmp37 < 512")
        tmp38 = tl.load(in_ptr3 + (r1 + (768*tmp37)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp34 + tmp38
        tmp40 = 768.0
        tmp41 = tmp12 / tmp40
        tmp42 = tmp39 - tmp41
        tmp43 = tmp28 / tmp40
        tmp44 = 1e-07
        tmp45 = tmp43 + tmp44
        tmp46 = tl.sqrt(tmp45)
        tmp47 = tmp42 / tmp46
        tmp48 = tmp30 * tmp47
        tmp50 = tmp48 + tmp49
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp50, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/li/cliadm33ljdyhpyebkvtnxly357mjiaabhdx5u3dzh2kcz3ksigm.py
# Source Nodes: [query_layer_1, query_layer_2, scale], Original ATen: [aten.add, aten.div, aten.sqrt]
# query_layer_1 => add_3
# query_layer_2 => div_1
# scale => full_default_1
triton_poi_fused_add_div_sqrt_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_sqrt_1', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x0 + (192*x2) + (2304*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 8.0
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpyylsfs5e5nft7erv2wjagizhhnlouqrtxj7v5jiui2ajocylom.py
# Source Nodes: [attention_probs, masked_fill_, output, rmask, tensor_1], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
# attention_probs => amax, div_2, exp, sub_2, sum_1
# masked_fill_ => full_default_4, where_1
# output => where
# rmask => full_default_2
# tensor_1 => full_default_3
triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 6144
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.full([1], False, tl.int1)
    tmp2 = -3.4028234663852886e+38
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tmp9 / tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp1, tmp15, tmp14)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp16, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkf3lsvdiybrz6mqmvgyeas77d5hvgaj3pdrtwfskhvbwuc2k7b.py
# Source Nodes: [value_layer_1], Original ATen: [aten.add]
# value_layer_1 => add_4
triton_poi_fused_add_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_3', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (192*x2) + (2304*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3i7xn4piox74tvyxyt65qsmfkfidcm3h3nakslqawckrx3v6x6q.py
# Source Nodes: [context_layer_1], Original ATen: [aten.clone]
# context_layer_1 => clone
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
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
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (32768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4llqphkhhvfowcpg2nxzp2wkpxvy4mtob2yk27vqfvuf65mijvz.py
# Source Nodes: [add_4, add_5, attention_output, hidden_states_6, mean_3, mul_4, pow_2, sqrt_2, sub_2, sub_3, variance_1], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
# add_4 => add_5
# add_5 => add_6
# attention_output => add_7
# hidden_states_6 => div_3
# mean_3 => mean_2
# mul_4 => mul_4
# pow_2 => pow_2
# sqrt_2 => sqrt_2
# sub_2 => sub_3
# sub_3 => sub_4
# variance_1 => mean_3
triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5', 'mutated_arg_names': []}
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
    tmp17 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = 768.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp4 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 / tmp9
    tmp19 = 1e-07
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sqrt(tmp20)
    tmp22 = tmp11 / tmp21
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgb2rmw3sgnwwiqksay3upzedqmcejma4gr5bqsle42dtzwx2gf.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_8, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/me/cmeief6nwnjfxhdq6n7v2x55wuq43ofhcg3rdkiduzwf24pox5t4.py
# Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
# start_logits_1 => clone_12
# start_loss => amax_12, exp_12, sub_62, sum_13
triton_per_fused__log_softmax_clone_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (2*r0), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp7, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfnn5u63jwnelvuyjivl64w24og6qci7l5a6icefl3c5zj5raml.py
# Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
# end_logits_1 => clone_13
# end_loss => amax_13, exp_13, sub_64, sum_16
triton_per_fused__log_softmax_clone_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (1 + (2*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (1))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp7, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpxm4myy7quaikdslhmkbaopj6up3oq4mosvyfneabnifc4yr7x.py
# Source Nodes: [add_98, end_loss, end_positions, start_loss, start_positions, total_loss], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_forward]
# add_98 => add_111
# end_loss => convert_element_type_13, div_50, full_default_52, ne_4, ne_5, neg_1, sum_17, sum_18, where_27
# end_positions => clamp_max_1, clamp_min_1
# start_loss => convert_element_type_12, div_49, full_default_50, ne_1, ne_2, neg, sum_14, sum_15, where_25
# start_positions => clamp_max, clamp_min
# total_loss => div_51
triton_poi_fused_add_clamp_div_nll_loss_forward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp12 = tl.load(in_out_ptr0 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp25 = tl.load(in_ptr3 + (0))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp35 = tl.load(in_ptr5 + (0))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp38 = tl.load(in_ptr6 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp7 = tl.where(tmp6, tmp5, tmp2)
    tmp8 = tmp7 + 512
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 512), "index out of bounds: 0 <= tmp10 < 512")
    tmp11 = tl.load(in_ptr1 + (tmp10), None, eviction_policy='evict_last')
    tmp14 = tmp11 - tmp13
    tmp17 = tl.log(tmp16)
    tmp18 = tmp14 - tmp17
    tmp19 = -tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp6.to(tl.int64)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp27 = triton_helpers.maximum(tmp26, tmp2)
    tmp28 = triton_helpers.minimum(tmp27, tmp4)
    tmp29 = tmp28 != tmp4
    tmp30 = tl.where(tmp29, tmp28, tmp2)
    tmp31 = tmp30 + 512
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert((0 <= tmp33) & (tmp33 < 512), "index out of bounds: 0 <= tmp33 < 512")
    tmp34 = tl.load(in_ptr4 + (tmp33), None, eviction_policy='evict_last')
    tmp37 = tmp34 - tmp36
    tmp40 = tl.log(tmp39)
    tmp41 = tmp37 - tmp40
    tmp42 = -tmp41
    tmp43 = tl.where(tmp29, tmp42, tmp20)
    tmp44 = tmp29.to(tl.int64)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp43 / tmp45
    tmp47 = tmp24 + tmp46
    tmp48 = 2.0
    tmp49 = tmp47 / tmp48
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp49, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1 = args
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
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
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
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (50265, 768), (768, 1))
    assert_size_stride(arg75_1, (512, 768), (768, 1))
    assert_size_stride(arg76_1, (2304, 768), (768, 1))
    assert_size_stride(arg77_1, (768, 768), (768, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (3072, ), (1, ))
    assert_size_stride(arg81_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (2304, 768), (768, 1))
    assert_size_stride(arg84_1, (768, 768), (768, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (3072, 768), (768, 1))
    assert_size_stride(arg87_1, (3072, ), (1, ))
    assert_size_stride(arg88_1, (768, 3072), (3072, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (2304, 768), (768, 1))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (3072, 768), (768, 1))
    assert_size_stride(arg94_1, (3072, ), (1, ))
    assert_size_stride(arg95_1, (768, 3072), (3072, 1))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (2304, 768), (768, 1))
    assert_size_stride(arg98_1, (768, 768), (768, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (3072, 768), (768, 1))
    assert_size_stride(arg101_1, (3072, ), (1, ))
    assert_size_stride(arg102_1, (768, 3072), (3072, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (2304, 768), (768, 1))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (3072, 768), (768, 1))
    assert_size_stride(arg108_1, (3072, ), (1, ))
    assert_size_stride(arg109_1, (768, 3072), (3072, 1))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (2304, 768), (768, 1))
    assert_size_stride(arg112_1, (768, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (3072, 768), (768, 1))
    assert_size_stride(arg115_1, (3072, ), (1, ))
    assert_size_stride(arg116_1, (768, 3072), (3072, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (2304, 768), (768, 1))
    assert_size_stride(arg119_1, (768, 768), (768, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (3072, 768), (768, 1))
    assert_size_stride(arg122_1, (3072, ), (1, ))
    assert_size_stride(arg123_1, (768, 3072), (3072, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (2304, 768), (768, 1))
    assert_size_stride(arg126_1, (768, 768), (768, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (3072, 768), (768, 1))
    assert_size_stride(arg129_1, (3072, ), (1, ))
    assert_size_stride(arg130_1, (768, 3072), (3072, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (2304, 768), (768, 1))
    assert_size_stride(arg133_1, (768, 768), (768, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (3072, 768), (768, 1))
    assert_size_stride(arg136_1, (3072, ), (1, ))
    assert_size_stride(arg137_1, (768, 3072), (3072, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (2304, 768), (768, 1))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (2304, 768), (768, 1))
    assert_size_stride(arg147_1, (768, 768), (768, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (3072, 768), (768, 1))
    assert_size_stride(arg150_1, (3072, ), (1, ))
    assert_size_stride(arg151_1, (768, 3072), (3072, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (2304, 768), (768, 1))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (3072, 768), (768, 1))
    assert_size_stride(arg157_1, (3072, ), (1, ))
    assert_size_stride(arg158_1, (768, 3072), (3072, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (2, 768), (768, 1))
    assert_size_stride(arg161_1, (2, ), (1, ))
    assert_size_stride(arg162_1, (1, 512), (512, 1))
    assert_size_stride(arg163_1, (1, 512), (512, 1))
    assert_size_stride(arg164_1, (1, ), (1, ))
    assert_size_stride(arg165_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embeddings, embeddings_1, hidden_states_1, mean, mul, position_embeddings, pow_1, query_states, sqrt, sub, sub_1, variance], Original ATen: [aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0.run(arg163_1, arg74_1, arg162_1, arg75_1, arg0_1, arg1_1, buf2, 512, 768, grid=grid(512), stream=stream0)
        del arg0_1
        del arg162_1
        del arg163_1
        del arg1_1
        del arg74_1
        del arg75_1
        buf3 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [qp], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (512, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 2304), (1, 768), 0), out=buf3)
        del arg76_1
        buf4 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [query_layer_1, query_layer_2, scale], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf3, arg2_1, buf4, 393216, grid=grid(393216), stream=stream0)
        del arg2_1
        buf5 = empty((12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf3, (12, 64, 512), (192, 1, 2304), 64), out=buf5)
        buf8 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, masked_fill_, output, rmask, tensor_1], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf5, buf8, 6144, 512, grid=grid(6144), stream=stream0)
        buf9 = buf4; del buf4  # reuse
        # Source Nodes: [value_layer_1], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf3, arg3_1, buf9, 393216, grid=grid(393216), stream=stream0)
        del arg3_1
        buf10 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf9, (12, 512, 64), (32768, 64, 1), 0), out=buf10)
        buf11 = reinterpret_tensor(buf9, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [context_layer_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf10, buf11, 393216, grid=grid(393216), stream=stream0)
        buf12 = reinterpret_tensor(buf10, (512, 768), (768, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (512, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), out=buf12)
        del arg77_1
        buf15 = reinterpret_tensor(buf11, (1, 512, 768), (393216, 768, 1), 0); del buf11  # reuse
        # Source Nodes: [add_4, add_5, attention_output, hidden_states_6, mean_3, mul_4, pow_2, sqrt_2, sub_2, sub_3, variance_1], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf12, arg78_1, buf2, arg4_1, arg5_1, buf15, 512, 768, grid=grid(512), stream=stream0)
        del arg4_1
        del arg5_1
        del arg78_1
        buf16 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (512, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), out=buf16)
        del arg79_1
        buf17 = reinterpret_tensor(buf16, (1, 512, 3072), (1572864, 3072, 1), 0); del buf16  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf17, arg80_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg80_1
        buf18 = reinterpret_tensor(buf2, (512, 768), (768, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), out=buf18)
        del arg81_1
        buf21 = reinterpret_tensor(buf12, (1, 512, 768), (393216, 768, 1), 0); del buf12  # reuse
        # Source Nodes: [add_7, add_8, hidden_states_14, mean_6, mul_5, pow_3, query_states_1, sqrt_3, sub_4, sub_5, variance_2], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf18, arg82_1, buf15, arg6_1, arg7_1, buf21, 512, 768, grid=grid(512), stream=stream0)
        del arg6_1
        del arg7_1
        del arg82_1
        buf22 = buf3; del buf3  # reuse
        # Source Nodes: [qp_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 2304), (1, 768), 0), out=buf22)
        del arg83_1
        buf23 = reinterpret_tensor(buf18, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf18  # reuse
        # Source Nodes: [query_layer_4, query_layer_5, scale_1], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf22, arg8_1, buf23, 393216, grid=grid(393216), stream=stream0)
        del arg8_1
        buf24 = reinterpret_tensor(buf8, (12, 512, 512), (262144, 512, 1), 0); del buf8  # reuse
        # Source Nodes: [attention_scores_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf22, (12, 64, 512), (192, 1, 2304), 64), out=buf24)
        buf27 = reinterpret_tensor(buf5, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf5  # reuse
        # Source Nodes: [attention_probs_2, masked_fill__1, output_2, rmask_1, tensor_3], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf24, buf27, 6144, 512, grid=grid(6144), stream=stream0)
        buf28 = buf23; del buf23  # reuse
        # Source Nodes: [value_layer_3], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf22, arg9_1, buf28, 393216, grid=grid(393216), stream=stream0)
        del arg9_1
        buf29 = reinterpret_tensor(buf15, (12, 512, 64), (32768, 64, 1), 0); del buf15  # reuse
        # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf28, (12, 512, 64), (32768, 64, 1), 0), out=buf29)
        buf30 = reinterpret_tensor(buf28, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [context_layer_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf29, buf30, 393216, grid=grid(393216), stream=stream0)
        buf31 = reinterpret_tensor(buf29, (512, 768), (768, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 768), (1, 768), 0), out=buf31)
        del arg84_1
        buf34 = reinterpret_tensor(buf30, (1, 512, 768), (393216, 768, 1), 0); del buf30  # reuse
        # Source Nodes: [add_12, add_13, attention_output_2, hidden_states_21, mean_9, mul_7, pow_4, sqrt_5, sub_6, sub_7, variance_3], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf31, arg85_1, buf21, arg10_1, arg11_1, buf34, 512, 768, grid=grid(512), stream=stream0)
        del arg10_1
        del arg11_1
        del arg85_1
        buf35 = reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (512, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 3072), (1, 768), 0), out=buf35)
        del arg86_1
        buf36 = reinterpret_tensor(buf35, (1, 512, 3072), (1572864, 3072, 1), 0); del buf35  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf36, arg87_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg87_1
        buf37 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg88_1, (3072, 768), (1, 3072), 0), out=buf37)
        del arg88_1
        buf40 = buf21; del buf21  # reuse
        # Source Nodes: [add_15, add_16, hidden_states_29, mean_12, mul_8, pow_5, query_states_2, sqrt_6, sub_8, sub_9, variance_4], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf37, arg89_1, buf34, arg12_1, arg13_1, buf40, 512, 768, grid=grid(512), stream=stream0)
        del arg12_1
        del arg13_1
        del arg89_1
        buf41 = buf22; del buf22  # reuse
        # Source Nodes: [qp_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (512, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 2304), (1, 768), 0), out=buf41)
        del arg90_1
        buf42 = reinterpret_tensor(buf37, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf37  # reuse
        # Source Nodes: [query_layer_7, query_layer_8, scale_2], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf41, arg14_1, buf42, 393216, grid=grid(393216), stream=stream0)
        del arg14_1
        buf43 = reinterpret_tensor(buf27, (12, 512, 512), (262144, 512, 1), 0); del buf27  # reuse
        # Source Nodes: [attention_scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf41, (12, 64, 512), (192, 1, 2304), 64), out=buf43)
        buf46 = reinterpret_tensor(buf24, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf24  # reuse
        # Source Nodes: [attention_probs_4, masked_fill__2, output_4, rmask_2, tensor_5], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf43, buf46, 6144, 512, grid=grid(6144), stream=stream0)
        buf47 = buf42; del buf42  # reuse
        # Source Nodes: [value_layer_5], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf41, arg15_1, buf47, 393216, grid=grid(393216), stream=stream0)
        del arg15_1
        buf48 = reinterpret_tensor(buf34, (12, 512, 64), (32768, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf47, (12, 512, 64), (32768, 64, 1), 0), out=buf48)
        buf49 = reinterpret_tensor(buf47, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf47  # reuse
        # Source Nodes: [context_layer_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf48, buf49, 393216, grid=grid(393216), stream=stream0)
        buf50 = reinterpret_tensor(buf48, (512, 768), (768, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (512, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), out=buf50)
        del arg91_1
        buf53 = reinterpret_tensor(buf49, (1, 512, 768), (393216, 768, 1), 0); del buf49  # reuse
        # Source Nodes: [add_20, add_21, attention_output_4, hidden_states_36, mean_15, mul_10, pow_6, sqrt_8, sub_10, sub_11, variance_5], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf50, arg92_1, buf40, arg16_1, arg17_1, buf53, 512, 768, grid=grid(512), stream=stream0)
        del arg16_1
        del arg17_1
        del arg92_1
        buf54 = reinterpret_tensor(buf36, (512, 3072), (3072, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 3072), (1, 768), 0), out=buf54)
        del arg93_1
        buf55 = reinterpret_tensor(buf54, (1, 512, 3072), (1572864, 3072, 1), 0); del buf54  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf55, arg94_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg94_1
        buf56 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg95_1, (3072, 768), (1, 3072), 0), out=buf56)
        del arg95_1
        buf59 = buf40; del buf40  # reuse
        # Source Nodes: [add_23, add_24, hidden_states_44, mean_18, mul_11, pow_7, query_states_3, sqrt_9, sub_12, sub_13, variance_6], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf56, arg96_1, buf53, arg18_1, arg19_1, buf59, 512, 768, grid=grid(512), stream=stream0)
        del arg18_1
        del arg19_1
        del arg96_1
        buf60 = buf41; del buf41  # reuse
        # Source Nodes: [qp_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 2304), (1, 768), 0), out=buf60)
        del arg97_1
        buf61 = reinterpret_tensor(buf56, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf56  # reuse
        # Source Nodes: [query_layer_10, query_layer_11, scale_3], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf60, arg20_1, buf61, 393216, grid=grid(393216), stream=stream0)
        del arg20_1
        buf62 = reinterpret_tensor(buf46, (12, 512, 512), (262144, 512, 1), 0); del buf46  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf61, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf60, (12, 64, 512), (192, 1, 2304), 64), out=buf62)
        buf65 = reinterpret_tensor(buf43, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf43  # reuse
        # Source Nodes: [attention_probs_6, masked_fill__3, output_6, rmask_3, tensor_7], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf62, buf65, 6144, 512, grid=grid(6144), stream=stream0)
        buf66 = buf61; del buf61  # reuse
        # Source Nodes: [value_layer_7], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf60, arg21_1, buf66, 393216, grid=grid(393216), stream=stream0)
        del arg21_1
        buf67 = reinterpret_tensor(buf53, (12, 512, 64), (32768, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf66, (12, 512, 64), (32768, 64, 1), 0), out=buf67)
        buf68 = reinterpret_tensor(buf66, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [context_layer_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf67, buf68, 393216, grid=grid(393216), stream=stream0)
        buf69 = reinterpret_tensor(buf67, (512, 768), (768, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (512, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 768), (1, 768), 0), out=buf69)
        del arg98_1
        buf72 = reinterpret_tensor(buf68, (1, 512, 768), (393216, 768, 1), 0); del buf68  # reuse
        # Source Nodes: [add_28, add_29, attention_output_6, hidden_states_51, mean_21, mul_13, pow_8, sqrt_11, sub_14, sub_15, variance_7], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf69, arg99_1, buf59, arg22_1, arg23_1, buf72, 512, 768, grid=grid(512), stream=stream0)
        del arg22_1
        del arg23_1
        del arg99_1
        buf73 = reinterpret_tensor(buf55, (512, 3072), (3072, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (512, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 3072), (1, 768), 0), out=buf73)
        del arg100_1
        buf74 = reinterpret_tensor(buf73, (1, 512, 3072), (1572864, 3072, 1), 0); del buf73  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf74, arg101_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg101_1
        buf75 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg102_1, (3072, 768), (1, 3072), 0), out=buf75)
        del arg102_1
        buf78 = buf59; del buf59  # reuse
        # Source Nodes: [add_31, add_32, hidden_states_59, mean_24, mul_14, pow_9, query_states_4, sqrt_12, sub_16, sub_17, variance_8], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf75, arg103_1, buf72, arg24_1, arg25_1, buf78, 512, 768, grid=grid(512), stream=stream0)
        del arg103_1
        del arg24_1
        del arg25_1
        buf79 = buf60; del buf60  # reuse
        # Source Nodes: [qp_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 2304), (1, 768), 0), out=buf79)
        del arg104_1
        buf80 = reinterpret_tensor(buf75, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [query_layer_13, query_layer_14, scale_4], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf79, arg26_1, buf80, 393216, grid=grid(393216), stream=stream0)
        del arg26_1
        buf81 = reinterpret_tensor(buf65, (12, 512, 512), (262144, 512, 1), 0); del buf65  # reuse
        # Source Nodes: [attention_scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf79, (12, 64, 512), (192, 1, 2304), 64), out=buf81)
        buf84 = reinterpret_tensor(buf62, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf62  # reuse
        # Source Nodes: [attention_probs_8, masked_fill__4, output_8, rmask_4, tensor_9], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf81, buf84, 6144, 512, grid=grid(6144), stream=stream0)
        buf85 = buf80; del buf80  # reuse
        # Source Nodes: [value_layer_9], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf79, arg27_1, buf85, 393216, grid=grid(393216), stream=stream0)
        del arg27_1
        buf86 = reinterpret_tensor(buf72, (12, 512, 64), (32768, 64, 1), 0); del buf72  # reuse
        # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf85, (12, 512, 64), (32768, 64, 1), 0), out=buf86)
        buf87 = reinterpret_tensor(buf85, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf85  # reuse
        # Source Nodes: [context_layer_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf86, buf87, 393216, grid=grid(393216), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (512, 768), (768, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf88)
        del arg105_1
        buf91 = reinterpret_tensor(buf87, (1, 512, 768), (393216, 768, 1), 0); del buf87  # reuse
        # Source Nodes: [add_36, add_37, attention_output_8, hidden_states_66, mean_27, mul_16, pow_10, sqrt_14, sub_18, sub_19, variance_9], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf88, arg106_1, buf78, arg28_1, arg29_1, buf91, 512, 768, grid=grid(512), stream=stream0)
        del arg106_1
        del arg28_1
        del arg29_1
        buf92 = reinterpret_tensor(buf74, (512, 3072), (3072, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (512, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 3072), (1, 768), 0), out=buf92)
        del arg107_1
        buf93 = reinterpret_tensor(buf92, (1, 512, 3072), (1572864, 3072, 1), 0); del buf92  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf93, arg108_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg108_1
        buf94 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg109_1, (3072, 768), (1, 3072), 0), out=buf94)
        del arg109_1
        buf97 = buf78; del buf78  # reuse
        # Source Nodes: [add_39, add_40, hidden_states_74, mean_30, mul_17, pow_11, query_states_5, sqrt_15, sub_20, sub_21, variance_10], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf94, arg110_1, buf91, arg30_1, arg31_1, buf97, 512, 768, grid=grid(512), stream=stream0)
        del arg110_1
        del arg30_1
        del arg31_1
        buf98 = buf79; del buf79  # reuse
        # Source Nodes: [qp_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (512, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 2304), (1, 768), 0), out=buf98)
        del arg111_1
        buf99 = reinterpret_tensor(buf94, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf94  # reuse
        # Source Nodes: [query_layer_16, query_layer_17, scale_5], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf98, arg32_1, buf99, 393216, grid=grid(393216), stream=stream0)
        del arg32_1
        buf100 = reinterpret_tensor(buf84, (12, 512, 512), (262144, 512, 1), 0); del buf84  # reuse
        # Source Nodes: [attention_scores_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf99, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf98, (12, 64, 512), (192, 1, 2304), 64), out=buf100)
        buf103 = reinterpret_tensor(buf81, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf81  # reuse
        # Source Nodes: [attention_probs_10, masked_fill__5, output_10, rmask_5, tensor_11], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf100, buf103, 6144, 512, grid=grid(6144), stream=stream0)
        buf104 = buf99; del buf99  # reuse
        # Source Nodes: [value_layer_11], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf98, arg33_1, buf104, 393216, grid=grid(393216), stream=stream0)
        del arg33_1
        buf105 = reinterpret_tensor(buf91, (12, 512, 64), (32768, 64, 1), 0); del buf91  # reuse
        # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf103, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf104, (12, 512, 64), (32768, 64, 1), 0), out=buf105)
        buf106 = reinterpret_tensor(buf104, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf104  # reuse
        # Source Nodes: [context_layer_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf105, buf106, 393216, grid=grid(393216), stream=stream0)
        buf107 = reinterpret_tensor(buf105, (512, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (512, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 768), (1, 768), 0), out=buf107)
        del arg112_1
        buf110 = reinterpret_tensor(buf106, (1, 512, 768), (393216, 768, 1), 0); del buf106  # reuse
        # Source Nodes: [add_44, add_45, attention_output_10, hidden_states_81, mean_33, mul_19, pow_12, sqrt_17, sub_22, sub_23, variance_11], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf107, arg113_1, buf97, arg34_1, arg35_1, buf110, 512, 768, grid=grid(512), stream=stream0)
        del arg113_1
        del arg34_1
        del arg35_1
        buf111 = reinterpret_tensor(buf93, (512, 3072), (3072, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (512, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 3072), (1, 768), 0), out=buf111)
        del arg114_1
        buf112 = reinterpret_tensor(buf111, (1, 512, 3072), (1572864, 3072, 1), 0); del buf111  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf112, arg115_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg115_1
        buf113 = reinterpret_tensor(buf97, (512, 768), (768, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf112, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg116_1, (3072, 768), (1, 3072), 0), out=buf113)
        del arg116_1
        buf116 = reinterpret_tensor(buf107, (1, 512, 768), (393216, 768, 1), 0); del buf107  # reuse
        # Source Nodes: [add_47, add_48, hidden_states_89, mean_36, mul_20, pow_13, query_states_6, sqrt_18, sub_24, sub_25, variance_12], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf113, arg117_1, buf110, arg36_1, arg37_1, buf116, 512, 768, grid=grid(512), stream=stream0)
        del arg117_1
        del arg36_1
        del arg37_1
        buf117 = buf98; del buf98  # reuse
        # Source Nodes: [qp_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (512, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 2304), (1, 768), 0), out=buf117)
        del arg118_1
        buf118 = reinterpret_tensor(buf113, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf113  # reuse
        # Source Nodes: [query_layer_19, query_layer_20, scale_6], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf117, arg38_1, buf118, 393216, grid=grid(393216), stream=stream0)
        del arg38_1
        buf119 = reinterpret_tensor(buf103, (12, 512, 512), (262144, 512, 1), 0); del buf103  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf118, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf117, (12, 64, 512), (192, 1, 2304), 64), out=buf119)
        buf122 = reinterpret_tensor(buf100, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf100  # reuse
        # Source Nodes: [attention_probs_12, masked_fill__6, output_12, rmask_6, tensor_13], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf119, buf122, 6144, 512, grid=grid(6144), stream=stream0)
        buf123 = buf118; del buf118  # reuse
        # Source Nodes: [value_layer_13], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf117, arg39_1, buf123, 393216, grid=grid(393216), stream=stream0)
        del arg39_1
        buf124 = reinterpret_tensor(buf110, (12, 512, 64), (32768, 64, 1), 0); del buf110  # reuse
        # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf123, (12, 512, 64), (32768, 64, 1), 0), out=buf124)
        buf125 = reinterpret_tensor(buf123, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf123  # reuse
        # Source Nodes: [context_layer_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf124, buf125, 393216, grid=grid(393216), stream=stream0)
        buf126 = reinterpret_tensor(buf124, (512, 768), (768, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), out=buf126)
        del arg119_1
        buf129 = reinterpret_tensor(buf125, (1, 512, 768), (393216, 768, 1), 0); del buf125  # reuse
        # Source Nodes: [add_52, add_53, attention_output_12, hidden_states_96, mean_39, mul_22, pow_14, sqrt_20, sub_26, sub_27, variance_13], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf126, arg120_1, buf116, arg40_1, arg41_1, buf129, 512, 768, grid=grid(512), stream=stream0)
        del arg120_1
        del arg40_1
        del arg41_1
        buf130 = reinterpret_tensor(buf112, (512, 3072), (3072, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (512, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 3072), (1, 768), 0), out=buf130)
        del arg121_1
        buf131 = reinterpret_tensor(buf130, (1, 512, 3072), (1572864, 3072, 1), 0); del buf130  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf131, arg122_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg122_1
        buf132 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg123_1, (3072, 768), (1, 3072), 0), out=buf132)
        del arg123_1
        buf135 = buf116; del buf116  # reuse
        # Source Nodes: [add_55, add_56, hidden_states_104, mean_42, mul_23, pow_15, query_states_7, sqrt_21, sub_28, sub_29, variance_14], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf132, arg124_1, buf129, arg42_1, arg43_1, buf135, 512, 768, grid=grid(512), stream=stream0)
        del arg124_1
        del arg42_1
        del arg43_1
        buf136 = buf117; del buf117  # reuse
        # Source Nodes: [qp_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (512, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 2304), (1, 768), 0), out=buf136)
        del arg125_1
        buf137 = reinterpret_tensor(buf132, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [query_layer_22, query_layer_23, scale_7], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf136, arg44_1, buf137, 393216, grid=grid(393216), stream=stream0)
        del arg44_1
        buf138 = reinterpret_tensor(buf122, (12, 512, 512), (262144, 512, 1), 0); del buf122  # reuse
        # Source Nodes: [attention_scores_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf136, (12, 64, 512), (192, 1, 2304), 64), out=buf138)
        buf141 = reinterpret_tensor(buf119, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf119  # reuse
        # Source Nodes: [attention_probs_14, masked_fill__7, output_14, rmask_7, tensor_15], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf138, buf141, 6144, 512, grid=grid(6144), stream=stream0)
        buf142 = buf137; del buf137  # reuse
        # Source Nodes: [value_layer_15], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf136, arg45_1, buf142, 393216, grid=grid(393216), stream=stream0)
        del arg45_1
        buf143 = reinterpret_tensor(buf129, (12, 512, 64), (32768, 64, 1), 0); del buf129  # reuse
        # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf142, (12, 512, 64), (32768, 64, 1), 0), out=buf143)
        buf144 = reinterpret_tensor(buf142, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf142  # reuse
        # Source Nodes: [context_layer_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf143, buf144, 393216, grid=grid(393216), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (512, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (512, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 768), (1, 768), 0), out=buf145)
        del arg126_1
        buf148 = reinterpret_tensor(buf144, (1, 512, 768), (393216, 768, 1), 0); del buf144  # reuse
        # Source Nodes: [add_60, add_61, attention_output_14, hidden_states_111, mean_45, mul_25, pow_16, sqrt_23, sub_30, sub_31, variance_15], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf145, arg127_1, buf135, arg46_1, arg47_1, buf148, 512, 768, grid=grid(512), stream=stream0)
        del arg127_1
        del arg46_1
        del arg47_1
        buf149 = reinterpret_tensor(buf131, (512, 3072), (3072, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 3072), (1, 768), 0), out=buf149)
        del arg128_1
        buf150 = reinterpret_tensor(buf149, (1, 512, 3072), (1572864, 3072, 1), 0); del buf149  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf150, arg129_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg129_1
        buf151 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg130_1, (3072, 768), (1, 3072), 0), out=buf151)
        del arg130_1
        buf154 = buf135; del buf135  # reuse
        # Source Nodes: [add_63, add_64, hidden_states_119, mean_48, mul_26, pow_17, query_states_8, sqrt_24, sub_32, sub_33, variance_16], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf151, arg131_1, buf148, arg48_1, arg49_1, buf154, 512, 768, grid=grid(512), stream=stream0)
        del arg131_1
        del arg48_1
        del arg49_1
        buf155 = buf136; del buf136  # reuse
        # Source Nodes: [qp_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (512, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 2304), (1, 768), 0), out=buf155)
        del arg132_1
        buf156 = reinterpret_tensor(buf151, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf151  # reuse
        # Source Nodes: [query_layer_25, query_layer_26, scale_8], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf155, arg50_1, buf156, 393216, grid=grid(393216), stream=stream0)
        del arg50_1
        buf157 = reinterpret_tensor(buf141, (12, 512, 512), (262144, 512, 1), 0); del buf141  # reuse
        # Source Nodes: [attention_scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf155, (12, 64, 512), (192, 1, 2304), 64), out=buf157)
        buf160 = reinterpret_tensor(buf138, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf138  # reuse
        # Source Nodes: [attention_probs_16, masked_fill__8, output_16, rmask_8, tensor_17], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf157, buf160, 6144, 512, grid=grid(6144), stream=stream0)
        buf161 = buf156; del buf156  # reuse
        # Source Nodes: [value_layer_17], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf155, arg51_1, buf161, 393216, grid=grid(393216), stream=stream0)
        del arg51_1
        buf162 = reinterpret_tensor(buf148, (12, 512, 64), (32768, 64, 1), 0); del buf148  # reuse
        # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf161, (12, 512, 64), (32768, 64, 1), 0), out=buf162)
        buf163 = reinterpret_tensor(buf161, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf161  # reuse
        # Source Nodes: [context_layer_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf162, buf163, 393216, grid=grid(393216), stream=stream0)
        buf164 = reinterpret_tensor(buf162, (512, 768), (768, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf163, (512, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 768), (1, 768), 0), out=buf164)
        del arg133_1
        buf167 = reinterpret_tensor(buf163, (1, 512, 768), (393216, 768, 1), 0); del buf163  # reuse
        # Source Nodes: [add_68, add_69, attention_output_16, hidden_states_126, mean_51, mul_28, pow_18, sqrt_26, sub_34, sub_35, variance_17], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf164, arg134_1, buf154, arg52_1, arg53_1, buf167, 512, 768, grid=grid(512), stream=stream0)
        del arg134_1
        del arg52_1
        del arg53_1
        buf168 = reinterpret_tensor(buf150, (512, 3072), (3072, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (512, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 3072), (1, 768), 0), out=buf168)
        del arg135_1
        buf169 = reinterpret_tensor(buf168, (1, 512, 3072), (1572864, 3072, 1), 0); del buf168  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf169, arg136_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg136_1
        buf170 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg137_1, (3072, 768), (1, 3072), 0), out=buf170)
        del arg137_1
        buf173 = buf154; del buf154  # reuse
        # Source Nodes: [add_71, add_72, hidden_states_134, mean_54, mul_29, pow_19, query_states_9, sqrt_27, sub_36, sub_37, variance_18], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf170, arg138_1, buf167, arg54_1, arg55_1, buf173, 512, 768, grid=grid(512), stream=stream0)
        del arg138_1
        del arg54_1
        del arg55_1
        buf174 = buf155; del buf155  # reuse
        # Source Nodes: [qp_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (512, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 2304), (1, 768), 0), out=buf174)
        del arg139_1
        buf175 = reinterpret_tensor(buf170, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [query_layer_28, query_layer_29, scale_9], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf174, arg56_1, buf175, 393216, grid=grid(393216), stream=stream0)
        del arg56_1
        buf176 = reinterpret_tensor(buf160, (12, 512, 512), (262144, 512, 1), 0); del buf160  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf175, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf174, (12, 64, 512), (192, 1, 2304), 64), out=buf176)
        buf179 = reinterpret_tensor(buf157, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf157  # reuse
        # Source Nodes: [attention_probs_18, masked_fill__9, output_18, rmask_9, tensor_19], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf176, buf179, 6144, 512, grid=grid(6144), stream=stream0)
        buf180 = buf175; del buf175  # reuse
        # Source Nodes: [value_layer_19], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf174, arg57_1, buf180, 393216, grid=grid(393216), stream=stream0)
        del arg57_1
        buf181 = reinterpret_tensor(buf167, (12, 512, 64), (32768, 64, 1), 0); del buf167  # reuse
        # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf179, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf180, (12, 512, 64), (32768, 64, 1), 0), out=buf181)
        buf182 = reinterpret_tensor(buf180, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf180  # reuse
        # Source Nodes: [context_layer_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf181, buf182, 393216, grid=grid(393216), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (512, 768), (768, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (512, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), out=buf183)
        del arg140_1
        buf186 = reinterpret_tensor(buf182, (1, 512, 768), (393216, 768, 1), 0); del buf182  # reuse
        # Source Nodes: [add_76, add_77, attention_output_18, hidden_states_141, mean_57, mul_31, pow_20, sqrt_29, sub_38, sub_39, variance_19], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf183, arg141_1, buf173, arg58_1, arg59_1, buf186, 512, 768, grid=grid(512), stream=stream0)
        del arg141_1
        del arg58_1
        del arg59_1
        buf187 = reinterpret_tensor(buf169, (512, 3072), (3072, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (512, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), out=buf187)
        del arg142_1
        buf188 = reinterpret_tensor(buf187, (1, 512, 3072), (1572864, 3072, 1), 0); del buf187  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf188, arg143_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg143_1
        buf189 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), out=buf189)
        del arg144_1
        buf192 = buf173; del buf173  # reuse
        # Source Nodes: [add_79, add_80, hidden_states_149, mean_60, mul_32, pow_21, query_states_10, sqrt_30, sub_40, sub_41, variance_20], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf189, arg145_1, buf186, arg60_1, arg61_1, buf192, 512, 768, grid=grid(512), stream=stream0)
        del arg145_1
        del arg60_1
        del arg61_1
        buf193 = buf174; del buf174  # reuse
        # Source Nodes: [qp_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (512, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 2304), (1, 768), 0), out=buf193)
        del arg146_1
        buf194 = reinterpret_tensor(buf189, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf189  # reuse
        # Source Nodes: [query_layer_31, query_layer_32, scale_10], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf193, arg62_1, buf194, 393216, grid=grid(393216), stream=stream0)
        del arg62_1
        buf195 = reinterpret_tensor(buf179, (12, 512, 512), (262144, 512, 1), 0); del buf179  # reuse
        # Source Nodes: [attention_scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf193, (12, 64, 512), (192, 1, 2304), 64), out=buf195)
        buf198 = reinterpret_tensor(buf176, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf176  # reuse
        # Source Nodes: [attention_probs_20, masked_fill__10, output_20, rmask_10, tensor_21], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf195, buf198, 6144, 512, grid=grid(6144), stream=stream0)
        buf199 = buf194; del buf194  # reuse
        # Source Nodes: [value_layer_21], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf193, arg63_1, buf199, 393216, grid=grid(393216), stream=stream0)
        del arg63_1
        buf200 = reinterpret_tensor(buf186, (12, 512, 64), (32768, 64, 1), 0); del buf186  # reuse
        # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf199, (12, 512, 64), (32768, 64, 1), 0), out=buf200)
        buf201 = reinterpret_tensor(buf199, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf199  # reuse
        # Source Nodes: [context_layer_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf200, buf201, 393216, grid=grid(393216), stream=stream0)
        buf202 = reinterpret_tensor(buf200, (512, 768), (768, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (512, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 768), (1, 768), 0), out=buf202)
        del arg147_1
        buf205 = reinterpret_tensor(buf201, (1, 512, 768), (393216, 768, 1), 0); del buf201  # reuse
        # Source Nodes: [add_84, add_85, attention_output_20, hidden_states_156, mean_63, mul_34, pow_22, sqrt_32, sub_42, sub_43, variance_21], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf202, arg148_1, buf192, arg64_1, arg65_1, buf205, 512, 768, grid=grid(512), stream=stream0)
        del arg148_1
        del arg64_1
        del arg65_1
        buf206 = reinterpret_tensor(buf188, (512, 3072), (3072, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (512, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 3072), (1, 768), 0), out=buf206)
        del arg149_1
        buf207 = reinterpret_tensor(buf206, (1, 512, 3072), (1572864, 3072, 1), 0); del buf206  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf207, arg150_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg150_1
        buf208 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg151_1, (3072, 768), (1, 3072), 0), out=buf208)
        del arg151_1
        buf211 = buf192; del buf192  # reuse
        # Source Nodes: [add_87, add_88, hidden_states_164, mean_66, mul_35, pow_23, query_states_11, sqrt_33, sub_44, sub_45, variance_22], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf208, arg152_1, buf205, arg66_1, arg67_1, buf211, 512, 768, grid=grid(512), stream=stream0)
        del arg152_1
        del arg66_1
        del arg67_1
        buf212 = buf193; del buf193  # reuse
        # Source Nodes: [qp_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 2304), (1, 768), 0), out=buf212)
        del arg153_1
        buf213 = reinterpret_tensor(buf208, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf208  # reuse
        # Source Nodes: [query_layer_34, query_layer_35, scale_11], Original ATen: [aten.add, aten.div, aten.sqrt]
        triton_poi_fused_add_div_sqrt_1.run(buf212, arg68_1, buf213, 393216, grid=grid(393216), stream=stream0)
        del arg68_1
        buf214 = reinterpret_tensor(buf198, (12, 512, 512), (262144, 512, 1), 0); del buf198  # reuse
        # Source Nodes: [attention_scores_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (12, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf212, (12, 64, 512), (192, 1, 2304), 64), out=buf214)
        buf217 = reinterpret_tensor(buf195, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf195  # reuse
        # Source Nodes: [attention_probs_22, masked_fill__11, output_22, rmask_11, tensor_23], Original ATen: [aten._softmax, aten.bitwise_not, aten.lift_fresh, aten.masked_fill]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_2.run(buf214, buf217, 6144, 512, grid=grid(6144), stream=stream0)
        del buf214
        buf218 = buf213; del buf213  # reuse
        # Source Nodes: [value_layer_23], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf212, arg69_1, buf218, 393216, grid=grid(393216), stream=stream0)
        del arg69_1
        del buf212
        buf219 = reinterpret_tensor(buf205, (12, 512, 64), (32768, 64, 1), 0); del buf205  # reuse
        # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf218, (12, 512, 64), (32768, 64, 1), 0), out=buf219)
        del buf217
        buf220 = reinterpret_tensor(buf218, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf218  # reuse
        # Source Nodes: [context_layer_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf219, buf220, 393216, grid=grid(393216), stream=stream0)
        buf221 = reinterpret_tensor(buf219, (512, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), out=buf221)
        del arg154_1
        buf224 = reinterpret_tensor(buf220, (1, 512, 768), (393216, 768, 1), 0); del buf220  # reuse
        # Source Nodes: [add_92, add_93, attention_output_22, hidden_states_171, mean_69, mul_37, pow_24, sqrt_35, sub_46, sub_47, variance_23], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf221, arg155_1, buf211, arg70_1, arg71_1, buf224, 512, 768, grid=grid(512), stream=stream0)
        del arg155_1
        del arg70_1
        del arg71_1
        buf225 = reinterpret_tensor(buf207, (512, 3072), (3072, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (512, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 3072), (1, 768), 0), out=buf225)
        del arg156_1
        buf226 = reinterpret_tensor(buf225, (1, 512, 3072), (1572864, 3072, 1), 0); del buf225  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf226, arg157_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg157_1
        buf227 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg158_1, (3072, 768), (1, 3072), 0), out=buf227)
        del arg158_1
        del buf226
        buf230 = buf211; del buf211  # reuse
        # Source Nodes: [add_95, add_96, hidden_states_179, mean_72, mul_38, pow_25, sequence_output, sqrt_36, sub_48, sub_49, variance_24], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
        triton_per_fused_add_div_mean_mul_pow_sqrt_sub_5.run(buf227, arg159_1, buf224, arg72_1, arg73_1, buf230, 512, 768, grid=grid(512), stream=stream0)
        del arg159_1
        del arg72_1
        del arg73_1
        del buf224
        del buf227
        buf231 = empty((512, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (512, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 2), (1, 768), 0), out=buf231)
        del arg160_1
        del buf230
        buf232 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf233 = empty((1, 1), device='cuda', dtype=torch.float32)
        buf234 = empty((1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_7.run(buf231, arg161_1, buf232, buf233, buf234, 1, 512, grid=grid(1), stream=stream0)
        buf235 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf236 = empty((1, 1), device='cuda', dtype=torch.float32)
        buf237 = empty((1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_8.run(buf231, arg161_1, buf235, buf236, buf237, 1, 512, grid=grid(1), stream=stream0)
        del arg161_1
        del buf231
        buf238 = reinterpret_tensor(buf233, (), (), 0); del buf233  # reuse
        buf239 = buf238; del buf238  # reuse
        # Source Nodes: [add_98, end_loss, end_positions, start_loss, start_positions, total_loss], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_forward]
        triton_poi_fused_add_clamp_div_nll_loss_forward_9.run(buf239, arg164_1, buf232, buf234, arg165_1, buf235, buf236, buf237, 1, grid=grid(1), stream=stream0)
        del arg164_1
        del arg165_1
        return (buf239, buf232, buf235, )


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
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg163_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg164_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg165_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForQuestionAnswering', benchmark_compiled_module)
